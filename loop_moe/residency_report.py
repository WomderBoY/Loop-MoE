"""Validate residency result invariants and produce a concise Chinese pilot report."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def verify(root):
    config = json.loads((root / 'residency_config.json').read_text())
    requests = read_rows(root / 'residency_requests.csv')
    paired = read_rows(root / 'paired_requests.csv')
    predictions = read_rows(root / 'prediction_requests.csv')
    expected = len(config['evaluation_request_ids']) * len(config['policies']) * len(config['budgets_percent'])
    if len(requests) != expected:
        raise ValueError(f'{root}: incomplete request results ({len(requests)} != {expected})')
    fit, validation, evaluation = (set(config[key]) for key in
                                 ('fit_request_ids', 'validation_request_ids', 'evaluation_request_ids'))
    if fit & validation or fit & evaluation or validation & evaluation:
        raise ValueError(f'{root}: overlapping split')
    lookup = {(row['budget_percent'], row['policy'], row['request_id']): row for row in requests}
    if len(lookup) != len(requests) or {r['request_id'] for r in requests} != evaluation:
        raise ValueError(f'{root}: duplicate or unexpected result IDs')
    for row in requests:
        misses = int(row['misses'])
        if int(row['h2d_bytes']) != misses * config['expert_bytes']:
            raise ValueError(f'{root}: byte accounting error')
        if config['memory_mode'] == 'peak' and int(row['peak_weight_bytes']) > int(row['budget_bytes']):
            raise ValueError(f'{root}: peak budget exceeded')
        oracle = lookup.get((row['budget_percent'], 'belady', row['request_id']))
        if oracle and misses < int(oracle['misses']):
            raise ValueError(f'{root}: policy beat exact offline oracle')
    for row in paired:
        actual = lookup[row['budget_percent'], row['policy'], row['request_id']]
        baseline = lookup[row['budget_percent'], row['baseline'], row['request_id']]
        saved = int(baseline['h2d_bytes']) - int(actual['h2d_bytes'])
        if saved != int(row['net_saved_h2d_bytes']) or saved != int(row['gained_hit_bytes'])-int(row['lost_hit_bytes']):
            raise ValueError(f'{root}: paired gains/losses do not reconcile')
        if actual['accesses'] != baseline['accesses']:
            raise ValueError(f'{root}: unequal service traces')
    for row in predictions:
        if not 0 <= int(row['hits']) <= min(int(row['selected']), int(row['eligible_targets']), int(row['targets'])):
            raise ValueError(f'{root}: invalid prediction recall counts')
    return config, read_rows(root / 'residency_summary.csv'), read_rows(root / 'paired_summary.csv')


def number(value):
    return f'{100 * float(value):.2f}%' if value != '' else '—'


def report(inputs, output):
    datasets = [(root, *verify(root)) for root in inputs]
    development = any(config["evaluation_role"].startswith("development") for _, config, _, _ in datasets)
    lines = ['# 下一次同池访问驻留：开发集回放' if development else '# 下一次同池访问驻留：回放报告', '',
             '本报告由结果文件自动生成，并校验峰值预算、Belady 下界、逐请求 H2D 与 gained−lost 收支。', '',
             ('使用已经查看过的 pilot 请求，结果只能作为开发对照。' if development else
              '使用显式请求划分；是否为未查看过的新请求由实验数据来源保证。') +
             '固定参数，无自动搜索；本模块只读结果文件。H2D 为轨迹推算流量，不是实测延迟。', '',
             '默认峰值预算包含正在传输/计算的一个专家；当前 gate 在服务前可见。与旧 replay 的预算和更新时序不同，旧数字不能直接相减。', '']
    for root, config, rows, pairs in datasets:
        model = config.get('model', root.name)
        focus = (5, 10) if model == 'Loopie-6B' else (10, 20) if model == 'Looped-MoE' else (5, 10, 20)
        lines += [f'## {model}', '',
                  f"{len(config['policies'])} 个策略 × {len(config['budgets_percent'])} 档预算；"
                  f"{len(config['fit_request_ids'])} 条拟合、{len(config['evaluation_request_ids'])} 条评估；"
                  f"专家大小 {config['expert_bytes']:,} 字节；{config['memory_mode']} 口径。", '',
                  '下表为按字节命中率。最强在线对照包含传统策略、gate/EMA 预测、合并/边界阶段及通用历史的两个版本，按整个评估集回顾性选出。', '',
                  '| 预算 | 最强在线对照 | 对照命中率 | 双历史，无折扣 | 双历史，距离折扣 | 通用历史，无折扣 | 通用历史，距离折扣 | Belady |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        lookup = {(float(r['budget_percent']), r['policy']): r for r in rows}
        for percent in config['budgets_percent']:
            if 'phase_ab_distance' not in config['policies']:
                continue
            main = lookup[float(percent), 'phase_ab_distance']
            reference = main['strongest_online_baseline']
            rates = [number(lookup[float(percent), p]['hit_rate_by_bytes']) for p in
                     (reference, 'phase_ab', 'phase_ab_distance', 'generic_lag', 'generic_lag_distance', 'belady')]
            lines.append(f'| {percent}% | {reference} | ' + ' | '.join(rates) + ' |')
        lines += ['', '重点预算的双历史＋距离折扣相对最强在线对照，正的净节省表示减少 H2D；标准差按请求计算：', '',
                  '| 预算 | 对照 | 新增命中 MiB | 损失命中 MiB | 净节省 MiB | 每位置 H2D 差值，均值 ± 标准差 MiB | 改善/退化请求 |',
                  '| --- | --- | ---: | ---: | ---: | ---: | --- |']
        for percent in focus:
            if (float(percent), 'phase_ab_distance') not in lookup:
                continue
            reference = lookup[float(percent), 'phase_ab_distance']['strongest_online_baseline']
            row = next(r for r in pairs if float(r['budget_percent']) == percent and
                       r['policy'] == 'phase_ab_distance' and r['baseline'] == reference)
            values = [float(row[key]) / 2**20 for key in ('gained_hit_bytes', 'lost_hit_bytes', 'net_saved_h2d_bytes',
                                                        'mean_delta_h2d_bytes_per_position', 'std_delta_h2d_bytes_per_position')]
            lines.append(f'| {percent}% | {reference} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:+.2f} | '
                         f'{values[3]:+.2f} ± {values[4]:.2f} | {row["requests_improved"]}/{row["requests_worsened"]} |')
        prediction = defaultdict(Counter)
        for row in read_rows(root / 'prediction_requests.csv'):
            budget = float(row['budget_percent']) if row['budget_percent'] else None
            prediction[row['predictor'], budget].update(hits=int(row['hits']), targets=int(row['targets']))
        audited = [percent for percent in focus if any(key[1] == percent for key in prediction)][:2]
        lines += ['', '预测诊断：完整池取目标 top-k；驻留候选统一来自 Score EMA 的相同集合，固定取 2 个。'
                  '以下驻留召回分母仍为全部目标专家；实际拟合 cell 数为各级表之和。', '',
                  '| 预测器 | 完整池 top-k 召回 | 拟合 cell 数 | ' +
                  ' | '.join(f'{percent}% 预算驻留召回' for percent in audited) + ' |',
                  '| --- | ---: | ---: | ' + ' | '.join('---:' for _ in audited) + ' |']
        for mode in config['fitted_table_cells']:
            metrics = [prediction[mode, budget] for budget in (None, *audited)]
            recalls = [number(str(counts['hits'] / counts['targets'])) if counts['targets'] else '—'
                       for counts in metrics]
            lines.append(f'| {mode} | {recalls[0]} | {sum(config["fitted_table_cells"][mode])} | ' +
                         ' | '.join(recalls[1:]) + ' |')
        lines += ['', f'原始输出：`{root}`。完整阶段消融、打乱对照、预测召回及每个生成位置的结果见同目录 CSV。', '']
    lines += ['## 解释范围', '',
              '需要同时比较分阶段双历史与相同历史跨度的通用历史，以及加/不加距离折扣。'
              '若主要收益也出现在通用历史版本，应把证据归于执行顺序或周期历史，不能声称已经证明分阶段 loop 信息的独立贡献。', '',
              '驻留候选召回统一使用 Score EMA 在 source 时刻的同一驻留集合。'
              'OLED 仅适配 gate 驻留与动态层配额，没有实现 CPU 补偿。统计表每级的容量上限相同，但实际 cell 数不同。'
              '首轮统计包含每个请求的冷启动；预测召回只统计存在实际后继的 source。', '',
              '最终结论还需要未查看的新请求、独立验证集及真实受限显存的端到端执行器。当前结果不支持将命中率直接换算成加速。', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Validated {len(inputs)} models; wrote {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path, help='Model result directories')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.results, args.output)


if __name__ == '__main__':
    main()
