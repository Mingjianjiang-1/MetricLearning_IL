from functools import partial
from utils import execute, env_paths
from multiprocessing import Pool

# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from train_bc_functions import train_bc_actor
from train_il_functions import train_il_actor
from train_gail_functions import train_gail_actor
from train_gmmil_functions import train_gmmil_actor
from train_sqil_functions import train_sqil_actor
from train_ilfo_functions import train_ilfo_actor
from train_ilbc_functions import train_ilbc_actor
from train_gmmilfo_functions import train_gmmilfo_actor
from train_gailfo_functions import train_gaifo_actor

def run_experiments(train_function, env, method_name, total_steps, ep, seed, alpha, args):
    env_name, h5path = env_paths[env]
    tasks = []
    # for seed in [1]:
    #     for ep in [16]:
    # suffix = ''
    # if method_name == 'ilbc':
    #     suffix = f'_{alpha}'
    if method_name == 'ilbc':
        path = f"logs/{env_name}/{method_name}_model_ep{ep}_seed{seed}_alpha{alpha}_{args.suffix}"
        tasks.append(
            partial(
                train_function,
                no_episodes=ep,
                env_name=env_name,
                seed=seed,
                h5path=h5path,
                total_steps=total_steps,
                file_path=path,
                alpha=alpha
            )
        )
    else:
        path = f"logs/{env_name}/{method_name}_model_ep{ep}_seed{seed}_{args.suffix}"
        tasks.append(
            partial(
                train_function,
                no_episodes=ep,
                env_name=env_name,
                seed=seed,
                h5path=h5path,
                total_steps=total_steps,
                file_path=path,
            )
        )

        # for ep in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        #     tasks.append(
        #         partial(
        #             train_function,
        #             no_episodes=1 / ep,
        #             env_name=env_name,
        #             seed=seed,
        #             h5path=h5path,
        #             total_steps=total_steps,
        #             file_path=f"{env_name}/{method_name}_model_ep1_{ep}_seed{seed}",
        #         )
        #     )

    with Pool(processes=60) as pool:
        print("QQ", tasks)
        actors = list(pool.map(execute, tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument("--method", type=str, default="bc")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--ep", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    train_functions_dict = {
        "bc": train_bc_actor,
        "il": train_il_actor,
        "gail": train_gail_actor,
        "gaifo": train_gaifo_actor,
        "gmmil": train_gmmil_actor,
        "sqil": train_sqil_actor,
        "ilfo": train_ilfo_actor,
        "ilbc": train_ilbc_actor,
        "gmmilfo": train_gmmilfo_actor
    }
    run_experiments(
            train_functions_dict[args.method], args.env, args.method, args.steps, args.ep, args.seed, alpha=args.alpha, args=args
        )


if __name__ == "__main__":
    main()
