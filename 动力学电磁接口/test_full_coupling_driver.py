"""Isolated mechanics/EM protocol tests, with explicitly synthetic callbacks."""
from dataclasses import replace
import unittest
from stepwise_em_model import DynamicState
from full_coupling_driver import predictor_corrector_full_step, FullCouplingConvergenceError


class FullCouplingTests(unittest.TestCase):
    def test_force_and_torque_converge_from_same_committed_memory(self):
        class SyntheticEM:
            def __init__(self):self.inputs=[]
            def trial_step(self,state,memory):
                self.inputs.append(memory)
                return {"Fmagnetic_xyz_N":(1.,0,0),"Tmagnetic_xyz_Nm":(0,.1,0)},{"trial":state.t_s}
        em=SyntheticEM();old={"t":0}
        seen=[]
        def dynamics(s,f,t,dt):
            seen.append((f,t));return replace(s,t_s=s.t_s+dt,omega_ball_y_rad_s=t[1]*dt)
        state=DynamicState(0,0,41,0,0,0,0)
        result=predictor_corrector_full_step(em,state,old,dynamics,.01,relaxation=1)
        self.assertEqual(result.iterations,2)
        self.assertEqual(result.state.omega_ball_y_rad_s,.001)
        self.assertTrue(all(x is old for x in em.inputs))
        self.assertEqual(old,{"t":0})

    def test_constant_force_alone_does_not_pass_varying_torque(self):
        class SyntheticEM:
            i=0
            def trial_step(self,state,memory):
                self.i+=1
                return {"Fmagnetic_xyz_N":(0,0,0),"Tmagnetic_xyz_Nm":(self.i,0,0)},memory
        state=DynamicState(0,0,41,0,0,0,0)
        with self.assertRaises(FullCouplingConvergenceError):
            predictor_corrector_full_step(SyntheticEM(),state,(),
                lambda s,f,t,dt:replace(s,t_s=s.t_s+dt),.01,max_iterations=2)


if __name__=="__main__":unittest.main()
