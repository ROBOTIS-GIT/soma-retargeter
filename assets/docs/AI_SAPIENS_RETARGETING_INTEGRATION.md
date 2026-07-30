# AI Sapiens Retargeting Integration

이 문서는 `/home/hc/work/src/open_src/soma-retargeter`에 AI Sapiens retargeting target을 추가한 변경 내용을 적용/검토할 수 있도록 정리한 기록이다.

작성 기준:

- 작업 대상 repository: `/home/hc/work/src/open_src/soma-retargeter`
- 공개 target 이름: `ai_sapiens`
- 기존 G1 target 이름: `unitree_g1`
- 품질 판단 범위: 이 문서는 실행/적용 구조만 설명한다. retargeting 품질, 초기 자세 일치, P1/P2 동등성은 별도 Evidence Gate 산출물 없이는 판단하지 않는다.

## 목적

기존 soma-retargeter는 기본적으로 SOMA BVH를 G1 CSV로 변환하는 구조였다. 이번 변경은 같은 실행 방식에서 AI Sapiens를 별도 target으로 선택할 수 있게 하는 것이 목적이다.

최종 사용자는 다음처럼 target만 `ai_sapiens`로 지정해 변환을 실행할 수 있어야 한다.

```bash
cd /home/hc/work/src/open_src/soma-retargeter
uv run python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_ai_sapiens_bvh_to_csv_converter_config.json \
  --viewer null
```

## 기존 G1 하드코딩과 AI Sapiens 적용 방식

원본 구조는 여러 로봇을 plugin처럼 자동 발견하는 구조가 아니었다. G1을 기준으로 실행 흐름 일부가 직접 고정되어 있었다.

확인된 G1 고정 지점:

```text
app/bvh_to_csv_converter.py
soma_retargeter/pipelines/utils.py
soma_retargeter/assets/csv.py
```

### 원본 G1 고정 방식

`app/bvh_to_csv_converter.py`의 viewer/model 로딩 경로는 G1일 때 Newton asset downloader를 직접 호출한다.

```python
newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"
```

즉 G1은 별도 MJCF 파일을 soma-retargeter repository 안에 넣지 않아도 Newton downloader를 통해 모델을 받을 수 있는 구조다.

또한 원본 converter는 기본 config에서 target을 G1로 지정한다.

```json
{
  "retarget_target": "unitree_g1"
}
```

CSV export도 기본적으로 G1 schema를 전제로 동작했다.

### AI Sapiens에 같은 구조를 적용하기 위해 필요한 변경

AI Sapiens는 Newton downloader에 기본 asset으로 등록된 G1과 같은 방식으로 받을 수 없기 때문에, `ROBOTIS-GIT/ai_sapiens` submodule의 URDF/STL asset을 참조하고 retargeting용 MJCF를 생성한다.

적용한 방식:

1. `TargetType.AI_SAPIENS`를 추가한다.
2. 문자열 target `ai_sapiens`를 parser에 등록한다.
3. `SOMA -> AI_SAPIENS` retargeter config lookup을 추가한다.
4. G1 downloader 대신 AI Sapiens MJCF resolver를 추가한다.
5. AI Sapiens MJCF와 mesh 참조를 submodule 기반으로 구성한다.
   - STL mesh 파일은 `third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1/`에서 가져온다.
   - retargeting용 MJCF는 submodule URDF를 floating-root MJCF로 compile한 뒤 proxy/TCP body patch를 적용해 생성한다.
6. AI Sapiens용 CSV schema를 추가한다.
7. 기존 G1 default config는 유지하고, AI Sapiens default config를 별도 파일로 추가한다.

이 방식은 G1 경로를 수정해서 바꾸는 방식이 아니라, G1과 같은 converter entry point에서 target만 바꿔 AI Sapiens를 선택할 수 있게 하는 방식이다.

```text
G1:
  retarget_target = unitree_g1
  model source = newton.utils.download_asset("unitree_g1")
  csv schema = UnitreeG1_CSVConfig

AI Sapiens:
  retarget_target = ai_sapiens
  model source = third_party/ai_sapiens URDF/STL + generated retarget MJCF
  mesh source = third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1
  mesh upload policy = STL files are tracked by the ai_sapiens submodule
  csv schema = AISapiens23DOF_CSVConfig
```

## 적용된 전체 구조

AI Sapiens는 G1을 덮어쓰지 않고 별도 target으로 추가했다.

```text
soma_retargeter/
  assets/
    ai_sapiens.py
    ai_sapiens_mjcf.py
    csv.py
  configs/
    ai_sapiens/
      ai_sapiens_retarget_mjcf_patch.json
      ai_sapiens_retarget.xml
      ai_sapiens_g1_aligned_feet_stabilizer_config.json
      soma_to_ai_sapiens_retargeter_config.json
      soma_to_ai_sapiens_g1_style_target_frame_scaler_config.json
  pipelines/
    newton_pipeline.py
    ik_objectives.py
    feet_stabilizer.py
    utils.py
  robotics/
    human_to_robot_scaler.py

assets/
  default_ai_sapiens_bvh_to_csv_converter_config.json
third_party/
  ai_sapiens/
    ai_sapiens_description/
      meshes/k1_rev1/
      urdf/k1_rev1/k1.urdf
tools/
  generate_ai_sapiens_retarget_mjcf.py
  compare_ai_sapiens_retarget_outputs.py
```

## 변경 파일 요약

### `app/bvh_to_csv_converter.py`

역할:

- 기존 `unitree_g1` 외에 `ai_sapiens` target을 받을 수 있게 했다.
- AI Sapiens retargeter config를 읽어 `NewtonPipeline`에 전달한다.
- AI Sapiens MJCF를 `soma_retargeter/assets/ai_sapiens.py`의 resolver로 찾는다.
- CSV export 시 target별 CSV schema를 사용한다.
- AI Sapiens root/output convention을 적용한다.

주요 동작:

1. `retarget_target` 값이 `ai_sapiens`이면 AI Sapiens 경로로 분기한다.
2. `retargeter_config`가 있으면 JSON을 읽어 pipeline에 넘긴다.
3. `ai_sapiens_mjcf`가 있으면 해당 MJCF를 로드한다.
4. target yaw 보정:
   - `ai_sapiens_target_position_yaw_deg`
   - `ai_sapiens_target_orientation_yaw_deg`
   - `ai_sapiens_target_yaw_pivot`
5. output root convention:
   - `ai_sapiens_root_translation_yaw_deg`
   - `ai_sapiens_root_orientation_yaw_deg`
   - `ai_sapiens_root_translation_xy_scale`
   - `ai_sapiens_ground_align`
6. viewer 초기 표시 보정:
   - `ai_sapiens_viewer_default_orientation_yaw_deg`

현재 공개 실행 설정은 `ai_sapiens_*` config key와 `SOMA_RETARGETER_AI_SAPIENS_*` environment variable만 읽는다.

이전 제품명 기반 fallback은 제거했다. 공개 실행 경로에는 AI Sapiens 이름만 남긴다.

### `soma_retargeter/pipelines/utils.py`

역할:

- `TargetType.AI_SAPIENS`를 추가했다.
- 문자열 target `ai_sapiens`를 enum으로 매핑한다.
- `SourceType.SOMA -> TargetType.AI_SAPIENS` 조합에서 AI Sapiens retargeter config를 반환한다.

적용 결과:

```python
get_target_type_from_str("ai_sapiens")
get_retargeter_config(SourceType.SOMA, TargetType.AI_SAPIENS)
```

두 경로가 동작한다.

### `soma_retargeter/assets/ai_sapiens.py`

역할:

- AI Sapiens MJCF 경로 resolver
- AI Sapiens joint name table
- AI Sapiens hand TCP local offset
- output root convention helper
- ground alignment helper

중요 공개 API:

```python
AI_SAPIENS_JOINT_NAMES
AI_SAPIENS_HAND_TCP_LOCAL
resolve_ai_sapiens_mjcf_path(...)
apply_ai_sapiens_root_convention(...)
compute_ground_alignment_offset(...)
apply_ground_alignment(...)
```

AI Sapiens asset helper는 `AI_SAPIENS_*` 이름과 `resolve_ai_sapiens_mjcf_path(...)`만 공개한다.

### `soma_retargeter/assets/csv.py`

역할:

- AI Sapiens 23DoF CSV schema를 추가했다.
- target 문자열에 따라 CSV schema를 선택하는 helper를 추가했다.

추가된 구조:

```python
AISapiens23DOF_CSVConfig
get_csv_config_for_target(target)
```

`get_csv_config_for_target("unitree_g1")`는 기존 G1 schema를 반환한다.

`get_csv_config_for_target("ai_sapiens")`는 AI Sapiens schema를 반환한다.

### `soma_retargeter/pipelines/newton_pipeline.py`

역할:

- AI Sapiens direct retargeting profile을 실행할 수 있는 Newton pipeline 경로를 추가했다.
- AI Sapiens MJCF를 직접 로드할 수 있게 했다.
- AI Sapiens용 target scaling, limb objective, staged body-chain solver, safety/step-limit 계열 설정을 읽는다.
- 공개 로그 라벨은 `AI Sapiens ...`로 출력되게 정리했다.

주의:

- 내부 변수명과 config key를 `ai_sapiens_*` 이름으로 정리했다.
- 공개 target 이름과 config 경로도 `ai_sapiens`를 사용한다.

즉 현재 구조는 다음과 같다.

```text
public target name: ai_sapiens
public config folder: soma_retargeter/configs/ai_sapiens
public mesh folder: third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1
internal tuning keys: ai_sapiens_*
```

### `soma_retargeter/pipelines/ik_objectives.py`

역할:

- AI Sapiens direct solver에서 사용하는 추가 IK objective 구현을 포함한다.
- limb bend, midpoint, source body frame preservation 등 AI Sapiens profile의 objective가 이 파일을 통해 연결된다.

### `soma_retargeter/pipelines/feet_stabilizer.py`

역할:

- AI Sapiens MJCF resolver를 사용할 수 있게 했다.
- `unitree_g1` 경로와 별도로 AI Sapiens model path를 해석할 수 있다.

### `soma_retargeter/robotics/human_to_robot_scaler.py`

역할:

- AI Sapiens scaler config에서 필요한 position offset mode와 yaw 기반 offset 변환을 처리할 수 있게 했다.
- 기존 G1 scaler 동작을 유지하면서 AI Sapiens config의 추가 key를 읽는다.

### `soma_retargeter/renderers/skeleton_renderer.py`

역할:

- Newton GL viewer에서 skeleton line 색상 인자를 안전하게 전달하도록 보정했다.
- 기존 renderer 구조는 유지하고, `log_lines(...)` 호출 시 색상을 tuple 형태로 넘긴다.

## 추가된 config 파일

### `assets/default_ai_sapiens_bvh_to_csv_converter_config.json`

AI Sapiens 예제 실행용 기본 config다.

핵심 값:

```json
{
  "import_folder": "assets/motions/bvh",
  "export_folder": "assets/motions/ai_sapiens-csv",
  "retargeter": "Newton",
  "retarget_source": "soma",
  "retarget_target": "ai_sapiens",
  "retargeter_config": "ai_sapiens/soma_to_ai_sapiens_retargeter_config.json",
  "ai_sapiens_mjcf": "ai_sapiens/ai_sapiens_retarget.xml",
  "ai_sapiens_viewer_default_orientation_yaw_deg": -90.0,
  "ai_sapiens_root_translation_yaw_deg": 0.0,
  "ai_sapiens_root_orientation_yaw_deg": 0.0,
  "ai_sapiens_root_translation_xy_scale": 1.0,
  "ai_sapiens_target_position_yaw_deg": 0.0,
  "ai_sapiens_target_orientation_yaw_deg": 0.0,
  "ai_sapiens_target_yaw_pivot": "origin",
  "ai_sapiens_ground_align": false
}
```

### `soma_retargeter/configs/ai_sapiens/soma_to_ai_sapiens_retargeter_config.json`

Newton pipeline에 전달되는 AI Sapiens retargeter profile이다.

핵심 값:

```json
{
  "robot_type": "ai_sapiens",
  "robot_mjcf": "ai_sapiens/ai_sapiens_retarget.xml",
  "human_robot_scaler_config": "ai_sapiens/soma_to_ai_sapiens_g1_style_target_frame_scaler_config.json"
}
```

이 파일에는 내부 solver tuning key가 많다. 해당 key도 `ai_sapiens_*` 이름을 사용한다.

### `soma_retargeter/configs/ai_sapiens/soma_to_ai_sapiens_g1_style_target_frame_scaler_config.json`

SOMA target body pose를 AI Sapiens solver target frame으로 변환하는 scaler config다.

역할:

- source body와 target body mapping
- target offset
- target yaw convention
- G1-style target frame alignment 기반 offset

### `soma_retargeter/configs/ai_sapiens/ai_sapiens_retarget.xml`

AI Sapiens retargeting용 MJCF다.

특징:

- mesh path는 `third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1`를 향한다.
- retargeting에서 필요한 virtual/proxy body를 포함한다.
- `tools/generate_ai_sapiens_retarget_mjcf.py`가 submodule URDF/STL과 patch JSON에서 생성한다.
- 생성 과정에서 `world -> pelvis` floating joint를 임시 URDF에 추가해 MuJoCo `nq=30`, `nv=29` 구조를 만든다.

### `soma_retargeter/configs/ai_sapiens/ai_sapiens_g1_aligned_feet_stabilizer_config.json`

feet stabilizer용 config다.

현재 기본 AI Sapiens converter config에서는 `post_processing`이 꺼져 있어 smoke 실행에서는 feet stabilizer가 활성 판단 대상이 아니다. 파일은 후처리 활성화 시 사용할 수 있도록 포함했다.

## mesh asset 정책

AI Sapiens MJCF는 submodule 내부의 아래 mesh 폴더를 참조한다.

```text
third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1/
```

현재 git 업로드 정책:

```text
STL 파일은 soma-retargeter repo에 직접 commit하지 않는다.
STL 파일은 ROBOTIS-GIT/ai_sapiens submodule에서 제공한다.
submodule은 고정 commit으로 관리한다.
```

MJCF의 `meshdir`는 다음 상대 경로를 사용한다.

```text
../../../third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1
```

따라서 실제 AI Sapiens model loading 또는 viewer 실행을 하려면 `git submodule update --init --recursive`로 submodule을 받아야 한다.

기존 로컬 STL 보존용 `.gitignore` 규칙은 남아 있지만 공식 입력 경로는 아니다.

```text
assets/robot_assets/ai_sapiens/meshes/*.stl
```

## 실행 방법

### AI Sapiens 전체 예제 변환

```bash
cd /home/hc/work/src/open_src/soma-retargeter
uv run python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_ai_sapiens_bvh_to_csv_converter_config.json \
  --viewer null
```

기본 입력:

```text
assets/motions/bvh
```

기본 출력:

```text
assets/motions/ai_sapiens-csv
```

### 기존 G1 예제 변환

기존 G1 config는 유지했다.

```bash
cd /home/hc/work/src/open_src/soma-retargeter
uv run python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_bvh_to_csv_converter_config.json \
  --viewer null
```

기본 출력:

```text
assets/motions/test-export
```

## 검증한 내용

아래 검증은 실행 가능성 검증이다. motion 품질 또는 초기 자세 일치 검증이 아니다.

### Python compile

명령:

```bash
cd /home/hc/work/src/open_src/soma-retargeter
uv run python -m compileall app soma_retargeter
```

결과:

```text
통과
```

### AI Sapiens config/MJCF 등록 확인

확인 내용:

- `get_target_type_from_str("ai_sapiens")`
- `get_retargeter_config(SourceType.SOMA, TargetType.AI_SAPIENS)`
- `resolve_ai_sapiens_mjcf_path(...)`
- AI Sapiens joint name count

확인 결과:

```text
target_type: AI_SAPIENS
retargeter_robot_type: ai_sapiens
retargeter_robot_mjcf: ai_sapiens/ai_sapiens_retarget.xml
joint_names: 23
```

### AI Sapiens full smoke

입력:

```text
assets/motions/bvh/high_jump_R_001__A277.bvh
```

실행 방식:

- 임시 입력 폴더에 BVH 1개만 복사
- `retarget_target: ai_sapiens`
- full frame 변환 실행

결과:

```text
CSV 생성 완료
high_jump_R_001__A277.csv
112732 bytes
```

### AI Sapiens 1-frame smoke

입력:

```text
assets/motions/bvh/high_jump_R_001__A277.bvh
```

실행 방식:

- 같은 BVH를 1-frame으로 잘라 임시 입력 생성
- `retarget_target: ai_sapiens`

결과:

```text
CSV 생성 완료
high_jump_R_001__A277.csv
1033 bytes
```

로그 확인:

```text
Target Robot Type: ai_sapiens
AI Sapiens Temporal Yaw/Twist Reference
AI Sapiens Direct Body Chain Staged Solver
AI Sapiens Limb Bend Angle Objective
```

공개 실행 로그는 `AI Sapiens` 라벨을 사용하도록 정리했다.

추가 확인:

```text
이전 제품명 기반 runtime fallback 없음
이전 제품명 기반 environment variable fallback 없음
이전 제품명 기반 asset alias 없음
이전 제품명 기반 MJCF resolver alias 없음
```

### 기존 G1 1-frame smoke

입력:

```text
assets/motions/bvh/high_jump_R_001__A277.bvh
```

실행 방식:

- 같은 BVH를 1-frame으로 잘라 임시 입력 생성
- `retarget_target: unitree_g1`

결과:

```text
CSV 생성 완료
high_jump_R_001__A277.csv
1247 bytes
```

이 검증은 기존 G1 경로가 최소 실행 경로에서 깨지지 않았음을 확인하기 위한 것이다.

### JSON validation

검증 파일:

```text
assets/default_ai_sapiens_bvh_to_csv_converter_config.json
soma_retargeter/configs/ai_sapiens/soma_to_ai_sapiens_retargeter_config.json
soma_retargeter/configs/ai_sapiens/soma_to_ai_sapiens_g1_style_target_frame_scaler_config.json
soma_retargeter/configs/ai_sapiens/ai_sapiens_g1_aligned_feet_stabilizer_config.json
```

결과:

```text
json_ok
```

### Instruction guard

명령:

```bash
python3 /home/hc/work/src/robotis_lab/scripts/tools/agent_instruction_guard.py
```

결과:

```text
agent_instruction_guard: OK (working)
```

## 현재 git 변경 상태

Tracked modified:

```text
.gitignore
app/bvh_to_csv_converter.py
soma_retargeter/assets/csv.py
soma_retargeter/pipelines/feet_stabilizer.py
soma_retargeter/pipelines/ik_objectives.py
soma_retargeter/pipelines/newton_pipeline.py
soma_retargeter/pipelines/utils.py
soma_retargeter/renderers/skeleton_renderer.py
soma_retargeter/robotics/human_to_robot_scaler.py
```

New files/directories:

```text
assets/default_ai_sapiens_bvh_to_csv_converter_config.json
soma_retargeter/assets/ai_sapiens.py
soma_retargeter/assets/ai_sapiens_mjcf.py
soma_retargeter/configs/ai_sapiens/
third_party/ai_sapiens/
tools/generate_ai_sapiens_retarget_mjcf.py
tools/compare_ai_sapiens_retarget_outputs.py
assets/docs/AI_SAPIENS_RETARGETING_INTEGRATION.md
```

Submodule에서 가져오는 폴더:

```text
third_party/ai_sapiens/ai_sapiens_description/urdf/k1_rev1/
third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1/
```

URDF와 STL 파일은 soma-retargeter repo가 직접 관리하지 않는다.

이미 repo에 존재하던 untracked output성 폴더:

```text
assets/motions/test-export/
assets/motions/test-export_20260610_152604/
```

위 두 폴더는 이번 문서/AI Sapiens 적용 작업에서 삭제하거나 정리하지 않았다.

## 품질 판단 제한

이 문서와 smoke test만으로는 아래 결론을 말할 수 없다.

- AI Sapiens retargeting 품질이 좋다.
- G1과 AI Sapiens 초기 자세가 일치한다.
- direct 결과가 via-G1 결과와 동등하다.
- 영상상 문제가 없다.

그 판단에는 별도 Evidence Gate가 필요하다.

초기 자세 일치를 말하려면 같은 motion directory 안에 아래 둘이 모두 있어야 한다.

```text
pipeline_via_g1.npz
pipeline_direct.npz
```

그리고 frame 0 기준으로 최소 아래 비교표가 필요하다.

```text
qpos
body position
body rotation
```

이 자료가 없으면 결론은 `판단 불가`로 둔다.

## 적용 시 확인할 것

적용자가 확인해야 하는 최소 항목:

1. `assets/default_ai_sapiens_bvh_to_csv_converter_config.json`가 포함되어 있는지 확인한다.
2. `soma_retargeter/configs/ai_sapiens/` 전체가 포함되어 있는지 확인한다.
3. `git submodule update --init --recursive` 후 `third_party/ai_sapiens`가 계획한 commit인지 확인한다.
4. `uv run python tools/generate_ai_sapiens_retarget_mjcf.py --check`를 실행한다.
5. `uv run python -m compileall app soma_retargeter tools`를 실행한다.
6. AI Sapiens 1개 BVH 변환을 실행해 CSV가 생성되는지 확인한다.
7. `uv run python tools/compare_ai_sapiens_retarget_outputs.py`로 `add-ai-sapiens-retargeting` branch의 기존 MJCF 결과와 정량 비교한다.
8. 기존 G1 config도 한 번 실행해 기존 경로가 깨지지 않았는지 확인한다.

## 이름 정리 상태

공개 target 이름, config folder, asset helper, model resolver, converter config, solver 내부 tuning key를 AI Sapiens 기준으로 정리했다.

최종 이름 체계:

```text
target string: ai_sapiens
config keys: ai_sapiens_*
environment variables: SOMA_RETARGETER_AI_SAPIENS_*
Python constants: AI_SAPIENS_*
MJCF resolver: resolve_ai_sapiens_mjcf_path(...)
config directory: soma_retargeter/configs/ai_sapiens
mesh directory: third_party/ai_sapiens/ai_sapiens_description/meshes/k1_rev1
```

제거 완료:

```text
이전 제품명 기반 runtime config fallback
이전 제품명 기반 environment variable fallback
이전 제품명 기반 asset alias
이전 제품명 기반 MJCF resolver alias
이전 제품명 기반 solver tuning key
이전 제품명 기반 solver attribute/function name
```

검증 명령:

```bash
rg -n -i "<이전 제품명 패턴>" . \
  --glob '!**/.git/**' \
  --glob '!**/.venv/**' \
  --glob '!**/__pycache__/**'
```

텍스트 파일 기준 검색 결과는 0건이다.
