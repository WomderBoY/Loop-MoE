"""Check persistent versus scratch lifetime without CUDA or model inference."""
import unittest
import torch
from loop_research.load_once_offload import SlotPlan,legacy_runtime,compare_logits


class TestSlots(unittest.TestCase):
    def test_diagnostics_separate_logits_from_decisions(self):
        a=torch.tensor([2.,1.,0.])
        same=compare_logits(a,a)
        self.assertEqual(same["max_abs"],0)
        self.assertEqual(same["top1_match"],1)
        offset=compare_logits(a,a+.5)
        self.assertEqual(offset["max_abs"],.5)
        self.assertEqual(offset["top1_match"],1)
        self.assertAlmostEqual(offset["kl"],0,places=6)
        flipped=compare_logits(a,torch.tensor([0.,3.,0.]))
        self.assertEqual(flipped["top1_match"],0)

    def test_archived_runtime_imports(self):
        self.assertEqual([c.__name__ for c in legacy_runtime()],
                         ["LoopedIncrementalEngine","FastPinnedExpertStore","FullResidentFastExecutor"])

    def test_retention_and_sequential_escapes(self):
        p=SlotPlan([(0,0),(1,0)],7,{0,1},True)
        self.assertEqual(p.locate(0,0,0,True),(0,False))
        self.assertEqual(p.locate(0,1,0,True),(2,True))
        self.assertEqual(p.locate(1,2,0,True),(3,True))
        self.assertEqual(p.locate(0,1,1,True),(2,False))
        self.assertEqual(p.locate(0,4,1,True),(6,True))
        self.assertEqual(p.locate(1,4,1,True),(6,True))
        self.assertNotIn((0,4),p.first)
        p.reset_token()
        self.assertEqual(p.locate(0,1,0,True),(2,True))

    def test_prefill_never_pins_union(self):
        p=SlotPlan([(0,0)],4,{0},True)
        for e in range(1,25):self.assertEqual(p.locate(0,e,0,False),(3,True))
        self.assertEqual(p.first,{})

    def test_static_does_not_retain_first(self):
        p=SlotPlan([(0,0),(0,1)],3,{0},False)
        for v in range(4):self.assertEqual(p.locate(0,2,v,True),(2,True))

    def test_peak_budget(self):
        with self.assertRaises(ValueError):SlotPlan([(0,0)],3,{0},True)
        p=SlotPlan([(0,0)],4,{0},True)
        self.assertEqual(p.locate(0,1,0,True),(1,True))
        self.assertEqual(p.locate(0,2,0,True),(2,True))
        with self.assertRaises(RuntimeError):p.locate(0,3,0,True)


if __name__=="__main__":unittest.main()
