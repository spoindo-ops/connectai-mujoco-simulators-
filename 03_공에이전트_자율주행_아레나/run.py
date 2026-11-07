import os
import time
import mujoco
import mujoco.viewer
import numpy as np

def main():
    xml_path = os.path.join(os.path.dirname(__file__), 'world.xml')
    with open(xml_path, 'r', encoding='utf-8') as f:
        model = mujoco.MjModel.from_xml_string(f.read())
    data = mujoco.MjData(model)

    print('=' * 60)
    print('MuJoCo 공 에이전트 자율주행 아레나 (Ball Navigation Arena)')
    print(' - 조작: Space(일시정지), R(리셋), 마우스 좌클릭(회전), 우클릭(이동)')
    print(' - 목표 지점(포털 기둥)으로 100kg 티타늄 볼이 주행하는 환경')
    print('=' * 60)

    data.qpos[0:3] = [-4.0, -4.0, 0.2]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            target = np.array([3.0, 3.0])
            pos = data.qpos[0:2]
            direction = target - pos
            dist = np.linalg.norm(direction)
            if dist > 0.45:
                direction = direction / dist
                data.ctrl[0] = float(np.clip(direction[0] * 50.0, -1.0, 1.0))
                data.ctrl[1] = float(np.clip(direction[1] * 50.0, -1.0, 1.0))
            else:
                data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == '__main__':
    main()
