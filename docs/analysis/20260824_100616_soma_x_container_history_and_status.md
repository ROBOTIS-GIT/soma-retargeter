# SOMA-X Container History and Current Status

## 작성 정보

- 작성 날짜와 시간: 2026-08-24 10:06:16 KST (Asia/Seoul)
- 작업 대상: `crazy_spence:/soma-retargeter`, `crazy_spence_mounted:/soma-retargeter`
- 브랜치/HEAD: 두 복사본 모두 `feature-phase2`, `80af132e8c60aaacc09212e9c61c1934fd52db0a`

## 문제 정의

두 컨테이너의 `/soma-retargeter`가 같은 bind mount인지, SOMA-X 변환 개발이 어느 복사본에서 진행됐는지, 실제 대량 변환이 어디까지 완료됐는지를 파일 이력과 산출물로 재구성한다.

## 원인 분석과 현재 증거

- `crazy_spence_mounted`는 2026-08-03 13:02:14 KST에 `docker commit crazy_spence soma-retargeter-local:folder-converter`로 만든 이미지에서 생성됐다.
- 두 컨테이너 모두 `/soma-retargeter`를 bind mount하지 않는다. 따라서 생성 시점에는 복사본이었고 이후 변경은 서로 공유되지 않았다.
- bind mount는 `crazy_spence_mounted`의 `/motion-input`과 `/motion-output`뿐이다.
- 2026-07-31의 초기 구현은 `smplx_to_soma.py`, `soma_x_npz_to_kimodo_npz.py`, AMASS 좌표 보정, 고정 180도 heading 보정의 다단계 실험이었다.
- 2026-08-03에 `soma_retargeter/assets/smplx_motion.py`와 `tools/convert_smplx_to_retarget_npz.py`로 Stage-II/AMASS -> SOMA77 변환이 통합됐다.
- 이미지 생성 이후 `crazy_spence_mounted`에서 anatomical heading canonicalization, frame-zero horizontal root rebase, 기존 출력 normalizer, Kimodo 22-joint matrix 입력, FPS override, `K0_5` 제외 기능이 추가됐다.
- `crazy_spence`에는 위 개선 전 converter가 남아 있고, 이후 K1 BVH/CSV batch resume 및 worker recycling 코드가 별도로 추가됐다.
- 변환 대상 67,281개는 원본 수가 아니었다. `K0_5` 밖 원본 12,853개, `K0_5` 일반 파생 33,556개, 심볼릭 링크 20,872개의 합계였다.
- 최종 원본 Stage-II 대상은 12,853개이며 출력은 12,852개다. 누락 파일은 `BMLmovi/Subject_49_F_MoSh/Subject_49_F_19_stageii.npz` 한 개로, ZIP 종료 레코드가 없는 5 MiB 절단 파일이다.
- `/motion-output`에는 별도 debug 파일 하나를 포함해 NPZ 12,853개, 약 95 GiB가 있다.

## 실제 적용된 개선 내용

- SOMA-X `PoseInversion(low_lod=True)` 기반 SMPL-X/AMASS -> SOMA77 변환 통합.
- 같은 model/gender/betas/device 조합의 runtime cache 재사용.
- frame-zero shoulder/hip anatomy로 forward를 계산해 Kimodo `+Z`로 heading 정규화.
- frame-zero root X/Z를 0으로 옮기고 모든 world position field에 동일 translation 적용.
- NPZ 내부 FPS를 보존하고 GUI config의 `kimodo_npz_fps`를 `null`로 설정.
- `(T,22,3,3)` Kimodo SMPL-X matrix 입력과 `(T,77,3,3)` 완성 SOMA77 입력을 관절 수로 구분.
- 기존 결과를 PoseInversion 재실행 없이 정규화하는 `tools/normalize_retarget_npz_heading.py` 추가.
- 재귀 변환에서 `--exclude-dir K0_5`를 사용해 파생 데이터와 심볼릭 링크를 제외.

## 현재 코드 복사본 차이

- `crazy_spence_mounted`가 SOMA-X 변환기의 최종 구현을 보유한다.
  - `smplx_motion.py` SHA-256: `0d80cb0ad2c40bd1f0a86ef55bc3dde819c4ce24370f040ac835148b63d6d92f`
  - converter SHA-256: `8234f5c98be0b2d8f77fb09dd3726930efc7b31859ab9550c7be675352afc76f`
  - normalizer SHA-256: `72474f30954306669207bdb63ed6ba399c7fda853229b903711f55c14f280bd7`
- `crazy_spence`는 구버전 변환기를 보유하고 normalizer가 없다.
  - `smplx_motion.py` SHA-256: `ef3a194ea4ae6894e5e4ecd525340ed2284d80689866744f3dcef0cae59307a7`
  - converter SHA-256: `2525846f8bd28e6bf420fdd9709764cb34e7be5e69e26a5847cf222dece0a144`
- 두 복사본의 SOMA-X 관련 변경은 모두 Git 미커밋 상태다. 현재 Git commit만으로 최종 변환기를 복원할 수 없다.

## 검증 명령

- `docker inspect crazy_spence crazy_spence_mounted`
- 두 컨테이너의 `git status --short`, `git log --all`, `sha256sum`, shell history 대조
- host input/output의 `find`, `wc -l`, `du -sh`, top-level dataset별 개수 비교
- `/motion-output` 12,852개 Stage-II NPZ의 metadata와 frame-zero `root_positions` 전수 검사

## 검증 결과

- Stage-II 출력 12,852개 모두 정상적으로 `np.load`됐다.
- `heading_canonicalized=True`: 12,852/12,852.
- `root_horizontal_rebased=True`: 12,852/12,852.
- frame-zero root X/Z 최대 절대값: `0`.
- FPS 분포: 60 Hz 532개, 100 Hz 4,581개, 120 Hz 7,682개, 250 Hz 57개.
- 6,899개 신규 변환에는 `source_coordinate=amass`가 있고, 기존 출력 후처리분 5,953개에는 해당 provenance field가 없다.
- `heading_canonicalized` 결과는 저장된 metadata의 전수 검사다. 12,852개 전체의 `posed_joints`에서 anatomical heading을 다시 계산하는 geometry 전수 검사는 이번 작업에서 수행하지 않았다.
- 현재 두 컨테이너 모두 SOMA-X 변환 프로세스는 실행 중이지 않다.

## 생성 산출물 경로

- 변환 결과: `/home/hc/work/src/robotis_lab_private/source/cyclo_lab/data/motions_soma`
- 본 분석 문서: `docs/analysis/20260824_100616_soma_x_container_history_and_status.md`
- 로컬 작업 로그: `.codex_local/work_logs/20260824_100616_soma_x_container_history_and_status.md`

## 남은 문제와 다음 작업

- 절단된 BMLmovi 원본 한 개는 정상 원본을 다시 확보해야 한다.
- 기존 출력 후처리분 5,953개는 정규화 metadata와 frame-zero root X/Z 검사는 통과했지만 `source_coordinate` provenance가 없고 실제 anatomical heading geometry는 이번 작업에서 전수 재계산하지 않았다.
- 최종 SOMA-X converter는 Git commit에 들어 있지 않고 `crazy_spence_mounted`에만 존재한다. 어느 repository/branch에 보존할지는 별도 사용자 지시가 필요하다.
- 이번 작업에서는 코드, 변환 산출물, 실행 프로세스, Git commit/push를 변경하지 않았다.
