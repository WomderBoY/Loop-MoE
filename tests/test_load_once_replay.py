import unittest
from types import SimpleNamespace
import torch
from loop_research.load_once_replay import RouterTape,tape_copy_count


class Router(torch.nn.Module):
    def forward(self,x,load_balancing_type=None):
        shape=(*x.shape[:-1],2)
        return torch.ones(shape),torch.zeros(shape,dtype=torch.long),torch.zeros(2,dtype=torch.long),None


class TestRouterTape(unittest.TestCase):
    def test_copy_count_includes_first_retention_and_escapes(self):
        def event(ids,slen):
            return (0,(1,slen,4),(torch.ones(1,1,2),torch.tensor([[ids]]),torch.zeros(4),None))
        events=[event([1,2],2),event([1,3],2),event([1,2],1),event([1,3],1)]
        self.assertEqual(tape_copy_count(events,[],3,{0},True,2),3)
        self.assertEqual(tape_copy_count(events,[],3,{0},False,2),4)

    def setUp(self):
        self.router=Router()
        self.model=SimpleNamespace(layers=[SimpleNamespace(moe=SimpleNamespace(router=self.router))])

    def test_record_is_owned_and_replay_restores_router(self):
        x=torch.zeros(1,1,4)
        tape=RouterTape(self.model)
        with tape.attached():original=self.router(x)
        original[0].zero_()
        self.assertEqual(float(tape.events[0][2][0].sum()),2)
        with RouterTape(self.model,tape.events).attached():got=self.router(x)
        self.assertTrue(torch.equal(got[0],torch.ones(1,1,2)))
        self.assertTrue(torch.equal(self.router(x)[0],torch.ones(1,1,2)))

    def test_wrong_shape_and_incomplete_tape_fail(self):
        tape=RouterTape(self.model)
        with tape.attached():self.router(torch.zeros(1,2,4))
        with self.assertRaises(RuntimeError):
            with RouterTape(self.model,tape.events).attached():self.router(torch.zeros(1,1,4))
        with self.assertRaises(RuntimeError):
            with RouterTape(self.model,tape.events).attached():pass

    def test_tape_cannot_supply_unrecorded_future_visits(self):
        tape=RouterTape(self.model)
        with tape.attached():self.router(torch.zeros(1,1,4))
        with self.assertRaises(RuntimeError):
            with RouterTape(self.model,tape.events).attached():
                self.router(torch.zeros(1,1,4));self.router(torch.zeros(1,1,4))


if __name__=="__main__":unittest.main()
