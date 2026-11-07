import os
import time
import mujoco
import mujoco.viewer

def main():
    xml_path = os.path.join(os.path.dirname(__file__), "world.xml")
    with open(xml_path, 'r', encoding='utf-8') as f:
        model = mujoco.MjModel.from_xml_string(f.read())
    data = mujoco.MjData(model)

    print("=" * 60)
    print("MuJoCo 1,000톤(1,000,000 kg) 초압축 강철공 시뮬레이터")
    print(" - 조작: Space(일시정지), R(리셋), 마우스 좌클릭(회전), 우클릭(이동)")
    print("=" * 60)

    last_drop = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)

            if time.time() - last_drop > 3.0:
                data.qpos[0:3] = [0, 0, 3.0]
                data.qvel[:] = 0
                last_drop = time.time()

            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
