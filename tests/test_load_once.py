"""Synthetic routing tests; these load no checkpoint and use no CUDA."""
import unittest
import torch
from loop_research.load_once import decide, make_hot


class TestLoadOnce(unittest.TestCase):
    def test_exact_first_and_zero_margin(self):
        s=torch.tensor([[[.1,.8,.6,.9]]])
        first=torch.tensor([[[0,2]]])
        hot=torch.tensor([True,False,True,False])
        for visit,tau in [(0,1e9),(1,0)]:
            got=decide(s,s,first,hot,mode="load_once",visit=visit,recurrent=True,tau=tau)
            self.assertTrue(torch.equal(got,s.topk(2,-1).indices))

    def test_hard_set_is_token_local(self):
        s=torch.tensor([[[.9,.8,.7,.6],[.9,.8,.7,.6]]])
        first=torch.tensor([[[2,3],[0,1]]])
        hot=torch.zeros(4,dtype=torch.bool)
        got=decide(s,s,first,hot,mode="load_once",visit=2,recurrent=True,tau=1e9)
        self.assertEqual(got.tolist(),[[[2,3],[0,1]]])
        first[0,1]=torch.tensor([2,3])
        got2=decide(s,s,first,hot,mode="load_once",visit=2,recurrent=True,tau=1e9)
        self.assertTrue(torch.equal(got[0,0],got2[0,0]))

    def test_escape_does_not_expand_first_set(self):
        s=torch.tensor([[[.9,.8,.4,.3]]])
        first=torch.tensor([[[2,3]]])
        before=first.clone()
        got=decide(s,s,first,torch.zeros(4,dtype=torch.bool),mode="load_once",visit=1,recurrent=True,tau=.5)
        self.assertEqual(got.tolist(),[[[0,1]]])
        self.assertTrue(torch.equal(first,before))

    def test_peak_budget_and_hot_only_floor(self):
        counts=torch.arange(700).reshape(28,25)
        for mode in ("native","load_once","hot_late","replay"):
            hot,transient=make_hot(counts,94,set(range(28)),mode)
            self.assertEqual(int(hot.sum())+transient+1,94)
            if mode=="load_once": self.assertEqual(transient,56)
            if mode=="hot_late": self.assertTrue(bool((hot.sum(-1)>=2).all()))
        with self.assertRaises(ValueError): make_hot(counts,50,set(range(28)),"load_once")

    def test_nonrecurrent_is_never_restricted(self):
        s=torch.tensor([[[.9,.8,.7,.6]]])
        got=decide(s,s,torch.tensor([[[2,3]]]),torch.zeros(4,dtype=torch.bool),mode="load_once",visit=1,recurrent=False,tau=1e9)
        self.assertEqual(got.tolist(),[[[0,1]]])


if __name__=="__main__": unittest.main()
