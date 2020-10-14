#  Copyright (c) 2020. Zoltan A Tuza, Guy-Bart Stan. All Rights Reserved.
#  This code is published under the MIT License.
#  Department of Bioengineering,
#  Centre for Synthetic Biology,
#  Imperial College London, London, UK
#  contact: ztuza@imperial.ac.uk, gstan@imperial.ac.uk

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ode_composer.signal_preprocessor import ZeroOrderHoldPreprocessor
from ode_composer.errors import ODEError
import copy
import numpy as np
from threading import Thread, Event
import time
import warnings
from statistics import mean
from itertools import chain, combinations

stop_event = Event()


def _watchdog(t, y, states, inputs=None):
    if stop_event.is_set():
        warnings.warn("Watchdog has been triggered!")
        return 0
    else:
        return 1


# this is need for the ODE solver to terminate when the event happens
_watchdog.terminal = True


class SolveStateSpaceModel(object):
    def __init__(self, ss_model, initial_conditions, tspan, inputs=None):
        self.ss_model = ss_model
        self.states = list(ss_model.state_vector.keys())
        self.init_cond = initial_conditions
        self.tspan = tspan
        self.inputs = inputs
        # Event object used to send signals from one thread to another
        self.sol = None

    def compute_solution(
        self, output_time=None, max_run_time=20, ode_config=None
    ):
        ode_config = SolveStateSpaceModel._check_ode_config(ode_config)
        action_thread = Thread(
            target=self._compute_solution, args=(output_time, ode_config)
        )

        # Here we start the thread and we wait 5 seconds before the code continues to execute.
        action_thread.start()
        action_thread.join(timeout=max_run_time)
        # We send a signal that tells the other thread to stop.
        stop_event.set()
        t = time.process_time()
        counter = 1
        while self.sol is None:
            print("waiting for the integrator to finish up")
            time.sleep(0.1)
            if counter > 1000:
                raise RuntimeError("No information from the ODE integrator")
            else:
                counter += 1
        elapsed_time = time.process_time() - t
        print(
            f"we have waited for the integrator to finish for: {elapsed_time} seconds"
        )
        stop_event.clear()
        return self.sol

    def _compute_solution(self, output_time=None, ode_config=None):
        t = time.process_time()
        if ode_config["use_jac"]:
            self.ss_model.compute_jacobian()
            jac = self.ss_model.get_jacobian
        else:
            jac = None

        self.sol = solve_ivp(
            fun=self.ss_model.get_rhs,
            t_span=self.tspan,
            y0=self.init_cond,
            args=(self.states, self.inputs),
            t_eval=output_time,
            jac=jac,
            events=_watchdog,
            method=ode_config["method"],
        )
        elapsed_time = time.process_time() - t
        print(f"elapsed time: {elapsed_time}")

    @staticmethod
    def _check_ode_config(ode_config):
        default_ode_config = {
            "method": "RK45",
            "use_jac": False,
            "show_time": False,
        }
        if ode_config is not None:
            default_ode_config.update(ode_config)
        return default_ode_config


class CompareStateSpaceWithData(object):
    def __init__(self, ss_model, db, exp_id, inputs=None):
        self.ss_model = ss_model
        self.states = list(ss_model.state_vector.keys())
        self.db = db
        self.exp_id = exp_id
        self.t = self.db.get_data(data_label="t", exp_id=self.exp_id)

        self.processed_inputs = None
        if inputs:
            self.processed_inputs = {}
            for one_input in inputs:
                u = db.get_data(data_label=one_input, exp_id=self.exp_id)
                u_preprocessor = ZeroOrderHoldPreprocessor(t=self.t, y=u)
                self.processed_inputs.update({one_input: u_preprocessor})

    def compute_solution(self, max_run_time=20, ode_config=None):
        tspan = [self.t.iloc[0], self.t.iloc[-1]]
        y0 = list(
            self.db.get_multicolumn_datum(
                data_labels=self.ss_model.state_vector.keys(),
                exp_id=self.exp_id,
                index=0,
            ).values()
        )

        self.ss_model_solver = SolveStateSpaceModel(
            ss_model=self.ss_model,
            initial_conditions=y0,
            tspan=tspan,
            inputs=self.processed_inputs,
        )
        self.ss_model_solution = self.ss_model_solver.compute_solution(
            output_time=self.t,
            max_run_time=max_run_time,
            ode_config=ode_config,
        )
        if self.ss_model_solution.status != 0:
            raise ValueError(
                f"ODE solution error: {self.ss_model_solution.message}"
            )

    def compute_solution_with_fixed_states(
        self, fixed_states, max_run_time=20
    ):
        tspan = [self.t.iloc[0], self.t.iloc[-1]]
        sub_ss_model = copy.deepcopy(self.ss_model)
        # we simulate one state, while the rest are supplied by measurement data (i.e. input to this state)
        merged_inputs = {}
        for one_state in fixed_states:
            # modify the state space model: remove fixed states
            sub_ss_model.state_vector.pop(one_state, None)
            data = self.db.get_data(data_label=one_state, exp_id=self.exp_id)
            preprocessor = ZeroOrderHoldPreprocessor(t=self.t, y=data)
            merged_inputs.update({one_state: preprocessor})
        # add the already existing inputs
        merged_inputs.update(self.processed_inputs)

        y0 = list(
            self.db.get_multicolumn_datum(
                data_labels=sub_ss_model.state_vector.keys(),
                exp_id=self.exp_id,
                index=0,
            ).values()
        )

        self.ss_model_solver = SolveStateSpaceModel(
            ss_model=sub_ss_model,
            initial_conditions=y0,
            tspan=tspan,
            inputs=merged_inputs,
        )
        self.ss_model_solution = self.ss_model_solver.compute_solution(
            output_time=self.t, max_run_time=max_run_time
        )
        if self.ss_model_solution.status != 0:
            raise ODEError(
                f"ODE solution error: {self.ss_model_solution.message}"
            )

    def plot_state(self, state_name, title=None, show_fig=True):
        data = self.db.get_data(data_label=state_name, exp_id=self.exp_id)
        idx = self.ss_model_solver.states.index(state_name)
        if show_fig:
            plt.figure()
        ax = plt.gca()
        ax.plot(self.t, data, label=f"$y{state_name[-1]}(t)$")
        ax.plot(
            self.ss_model_solution.t,
            self.ss_model_solution.y[idx, :],
            label=f"${state_name}(t)$",
        )
        if title:
            ax.title.set_text(title)
        else:
            state_aic = self.compute_aic_for_state(state_name=state_name)
            state_residual = self.compute_residual_for_state(
                state_name=state_name
            )
            ax.set_title(
                f"AIC: {state_aic:.3g} Residual: {state_residual:.3g}"
            )

        plt.legend(loc="best")
        ax.set_xlabel("time [min]")
        ax.set_ylabel("y value")
        plt.grid()
        if show_fig:
            plt.show()

    def plot_model(self, model_name="", file_name=None):
        plt.figure()
        model_aic = self.compute_aic_for_model()
        model_residual = self.compute_residual_for_model()
        model_performance = (
            f" AIC: {model_aic:.3g} Residual: {model_residual:.3g}"
        )
        plt.tight_layout()
        plt.suptitle(model_name + model_performance, fontsize=12)
        plt.subplot(321)
        self.plot_state(state_name=f"x{1}", show_fig=False)
        plt.subplot(322)
        self.plot_state(state_name=f"x{2}", show_fig=False)
        plt.subplot(325)
        self.plot_state(state_name=f"x{3}", show_fig=False)
        plt.subplot(326)
        self.plot_state(state_name=f"x{4}", show_fig=False)
        if file_name:
            plt.savefig(file_name)
        else:
            plt.show()

    def compute_residual_for_state(self, state_name) -> float:
        data = self.db.get_data(data_label=state_name, exp_id=self.exp_id)
        idx = self.ss_model_solver.states.index(state_name)
        one_state = self.ss_model_solution.y[idx, :]
        res = (np.sqrt(np.square(data - one_state).sum())) / (2 * len(data))
        return res

    def compute_aic_for_state(self, state_name):
        """

        Args:
            state_name: name of the state in the model

        Returns: the AIC of the selected state in the model

        """
        return 2 * self.compute_residual_for_state(
            state_name=state_name
        ) + 2 * len(self.ss_model.state_vector[state_name])

    def compute_residual_for_model(self):
        return mean(
            [
                self.compute_residual_for_state(state_name=state_name)
                for state_name in self.states
            ]
        )

    def compute_aic_for_model(self):
        return 2 * self.compute_residual_for_model() + 2 * sum(
            [
                self.get_num_rhs_for_state(state_name)
                for state_name in self.states
            ]
        )

    def get_num_rhs_for_state(self, state_name):
        if state_name not in self.ss_model.state_vector:
            raise ValueError(f"state {state_name} is not in the model")
        return len(self.ss_model.state_vector[state_name])

    def compute_solution_with_fixed_states_combinations(self, max_run_time=20):
        # how many states we have
        state_num = len(self.states)
        # compute all possible combinations
        all_combo = [
            combinations(range(1, state_num + 1), x)
            for x in reversed(range(state_num))
        ]
        results = dict()
        for combi in chain(*all_combo):
            fixed_states = [f"x{num}" for num in combi]
            print(f"running with fixed states: {fixed_states}\n")
            aic = None
            try:
                self.compute_solution_with_fixed_states(
                    fixed_states=fixed_states, max_run_time=max_run_time
                )
                # because we have partially fixed model, we need to calculate the aic manually
                # get the dynamic states
                dynamic_states = list(
                    set(self.states) - (set(self.states) & set(fixed_states))
                )
                aic = 0.0
                residuals = []
                rhs_terms = 0
                for dyn_state in dynamic_states:
                    residuals.append(
                        self.compute_residual_for_state(state_name=dyn_state)
                    )
                    rhs_terms += self.get_num_rhs_for_state(
                        state_name=dyn_state
                    )

                aic += 2 * mean(residuals) + 2 * rhs_terms
            except ODEError as e:
                print(e)
                aic = None
            finally:
                results.update({tuple(combi): aic})

        # report the results
        return results
