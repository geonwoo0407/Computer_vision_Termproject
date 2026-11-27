# SeatSense Vision Module

도서관 좌석 관측을 위한 RGB-D Vision 모듈입니다. RGB 영상과 RealSense
RGB-D 입력, YOLO 객체 검출, 객체별 거리/카메라 좌표, 좌석 ROI 후보,
바운딩 박스와 FPS 표시, JSONL 출력까지 구현되어 있습니다. Tracking ID,
사람-소지품 소유 관계, 최종 좌석 상태는 이 모듈에서 결정하지 않습니다.

## PC 환경 준비

Python 3.10 이상을 권장합니다. 프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
python -m pip install -r requirements.txt
```

이 설치 방법은 현재 x86_64 개발 PC 기준입니다. Jetson에서는 JetPack/L4T를
먼저 확인하고 NVIDIA가 제공하는 호환 PyTorch를 사용해야 하므로 위 명령을
그대로 실행하지 않습니다.

Linux에서 CUDA 패키지 없이 CPU 전용으로 가볍게 설치하려면 PyTorch를 먼저
설치한 뒤 나머지 의존성을 설치합니다.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

기본 모델은 가벼운 `yolo11n.pt`이며 첫 실행 때 Ultralytics가 모델 파일을
다운로드합니다. 기본 장치는 CPU입니다. CUDA가 설정된 PyTorch 환경에서는
`--device 0`으로 GPU를 선택할 수 있습니다.

## 실행

`main.py`의 검출·카메라 테스트와 `multiview.py live/demo`는 실행할 때마다 `output/logs/`에 새 JSONL 파일을 만들고 터미널에 경로를 표시합니다. 좌석 모드는 빈자리 기준 촬영 후부터 처리한 프레임을 기록합니다. YOLO 로그에는 카메라 ID와 시리얼, 프레임 번호·시각, 모델과 임계값, 검출 클래스·신뢰도·박스가 들어갑니다. 여섯 자리 모드에서는 `seat_grid`에 자리별 판단도 남습니다. `--jsonl 경로`를 지정하면 해당 경로를 사용하고, 같은 이름이 이미 있으면 같은 폴더에 고유한 이름으로 새 파일을 만듭니다.

### 카메라 한 대에서 3열 × 2행 관측하기

```bash
python main.py --six-seats
```

화면의 먼 줄은 왼쪽부터 **1 · 2 · 3**, 가까운 줄은 **4 · 5 · 6**입니다. 번호는 카메라 화면 안에서만 쓰는 임시 번호입니다.

1. 여섯 상판과 의자가 보이도록 카메라를 고정하고 자리를 비운 뒤 **F**로 화면을 고정합니다.
2. 안내되는 자리마다 **상판 → 의자와 착석자 몸통 구역** 순서로 드래그하고 각각 **Enter**로 확정합니다. 총 12개 영역입니다. **C**로 이전 선택을 다시 할 수 있습니다. 상판은 칸막이 안쪽의 보이는 면을 잡고, 의자는 그 책상의 것만 포함합니다.
3. 약 3초간 여섯 자리를 비워 두면 빈자리 기준 영상을 만들고 관측을 시작합니다. 영상 아래 2행 3열 표에 자리별 상태, 근거, 지지 프레임 수와 관찰 시간이 표시됩니다. 사용중·자리비움·미사용중 전환에는 한 자리 모드와 같은 최소 5초 / 80% / 8프레임 기준을 적용합니다. **Q**로 종료합니다.

영역은 `output/seat_grid_3x2.json`에 저장됩니다. 다음 실행에서 **F**는 저장된 영역을 사용하고 **R**은 현재 구도로 다시 등록합니다. 빈자리 기준 영상은 실행할 때마다 새로 촬영합니다. 다른 구도를 따로 저장하려면 `--grid-layout output/다른구도.json`을 지정합니다. `--jsonl output/grid_observations.jsonl`을 추가하면 각 프레임의 `seat_grid` 배열에 여섯 자리의 번호, 행·열, 상태 및 판정 근거가 기록됩니다.

**상태 경과**는 해당 상태를 처음 확정하거나 다른 상태로 전환한 시점부터 셉니다. 첫 확정 전에는 `--`이며, 관측 보류 중에도 계속 흐릅니다. **후보**는 현재 후보를 추적한 시간입니다. 로그에서 각각 `state_elapsed_seconds`와 `observed_seconds`로 확인할 수 있습니다.

여섯 자리를 앱에 연동하려면 관리자 화면에서 **시연 테스트 → 카메라 API 모드**를 선택합니다. 현재 앱 좌석 번호는 먼 줄 **127 · 26 · 25**, 가까운 줄 **22 · 23 · 24**입니다. 다른 자리에서 시연할 때는 같은 순서로 `--app-seats` 뒤에 여섯 번호를 지정합니다.

```bash
python main.py --six-seats --app-url https://hello-flask-seat.onrender.com
```

실행 시 터미널에 앱 관리자 비밀번호를 입력합니다. 자동 실행할 때는 환경 변수 `SEAT_ADMIN_PASSWORD`를 사용할 수 있습니다. 로컬 앱에서는 `--app-url`에 실행 중인 로컬 서버 주소를 넣습니다. 터미널에 `SeatSense updated seat`가 뜨면 서버가 해당 좌석 관측을 수신한 것입니다. 앱에는 확인된 사람 또는 짐 유무를 보내며, 보류 중에는 마지막으로 보낸 관측을 유지합니다.

### 카메라 한 대에서 한 좌석의 상태 보기

```bash
python main.py --single-seat
```

카메라를 고정하고 책상과 의자를 비웁니다. 미리보기에서 **F**를 누른 뒤 한 좌석의 **책상 상판**을 드래그하고 Enter, 같은 좌석의 **의자와 앉은 사람의 몸통이 보일 구역**을 드래그하고 Enter를 누릅니다. 약 3초 동안 그대로 두면 빈자리 기준 영상을 만들고 판정을 시작합니다. **Q**로 종료합니다. 카메라 두 대가 연결되어 있을 때는 `--serial 시리얼번호`로 한 대를 고릅니다.

| 화면 상태 | 해당 좌석에서 확인하는 증거 |
|---|---|
| 사용중 | 의자 구역에서 사람 몸통이 계속 검출됨 |
| 자리비움 | 사람이 보이지 않고, 책상 또는 의자에 COCO 소지품이나 국소적인 RGB 변화가 반복됨 |
| 미사용중 | 사람과 짐 후보가 없고, 책상과 의자가 빈 기준과 일치함 |

좌석 상태는 기본 **최소 5초**, 최근 5초간 처리한 프레임의 **80% 이상**, **최소 8프레임**이 같은 후보를 지지할 때 전환됩니다. 객체는 겹치는 위치에서 추적하며 관측이 1초 넘게 끊기면 후보 시간을 다시 셉니다. 가장 최근에 확정한 상태가 화면 하단에 표시됩니다. `Hits`는 지지 프레임 수 / 처리 프레임 수, `Candidate` 옆의 초는 후보 관찰 시간, YOLO 박스의 소수는 해당 프레임의 모델 신뢰도입니다. 주황색 `RGB` 박스는 빈자리 영상과의 차이로 얻은 짐 후보입니다. 국소적인 짙은 그림자와 소지품은 한 시점의 RGB만으로 구분하기 어려울 수 있습니다.

이 모드는 RGB로만 동작합니다. 자리비움은 시각적 관측이며 예약자 확인과 자동반납은 별도 단계입니다. 기준 의자가 검출되면 현재 의자 위치와 크기에 맞춰 내부를 비교하고, 시야가 크게 가려진 구간은 상태 전환을 보류합니다. 더 오래 보려면 `--confirm-seconds 8`처럼 지정합니다. `--jsonl output/single_seat.jsonl`을 추가하면 프레임별 상태, 후보, 신뢰도와 표 수를 저장합니다. 같은 구도의 영상 파일로 재현할 때는 `--input-type video --source 파일경로`를 추가하고, 무화면 실행에는 `--desk-roi X Y W H --chair-roi X Y W H --no-display`를 함께 지정합니다.

### 체커보드로 두 RGB 카메라의 3D 좌표계 맞추기

PC의 OpenCV, NumPy, pyrealsense2와 Matplotlib을 사용합니다. 두 D435/D435i를
USB 3 포트에 연결하고 앞뒤 카메라를 최종 위치에 고정합니다. 같은 프레임에서
두 카메라에 보드의 **같은 인쇄면 전체**가 보여야 합니다. 마주 보는 구도에서는
둘 다 내려다보는 책상 위에 놓고 각도를 바꿔 보세요. 반대 면에 별도로 인쇄한
체커보드는 두께와 원점 차이가 생기므로 이 절차에 쓸 수 없습니다.

1. **보드 준비.** A4 가로 PDF를 100% / 실제 크기로 인쇄해 평평한 판에
   붙입니다. 기본은 가로 10칸 × 세로 7칸에 **내부 코너 9×6**입니다.
   여러 칸의 길이를 자로 재고 칸 수로 나눈 실측값을 사용합니다.
2. **촬영.** 아래 25 mm는 예시입니다. 첫 시리얼이 공통 RGB 좌표계가 됩니다.
   `--serials`를 생략하면 정확히 두 대가 연결된 경우 시리얼 정렬 순서로
   지정하고 터미널에 표시합니다. 해상도·FPS는 `config.json`의 RealSense
   항목, 패턴·칸 크기·목표 쌍 수는 `checkerboard` 또는 CLI로 지정합니다.

```bash
python stereo_board.py pattern output/board.pdf --inner 9 6 --square-mm 25
python main.py --list-cameras
python stereo_board.py capture output/stereo.json \
  --serials 948522070995 148522071908 --inner 9 6 --square-mm 25 --pairs 20
```

3. RGB/정렬 Depth 창에서 보드를 정지시키고 **F**로 한 쌍을 고정해 코너를
   찾습니다. 좌우 화면에서 A(0번)와 B(첫 행의 끝)가 같은 실제 점인지
   확인합니다. **1 / 2**로 해당 카메라의 번호 방향을 바꾸고 **Enter**로
   채택합니다. **C**는 현재 쌍 취소, **X**는 수집 목록 초기화입니다.
   화면 중앙·가장자리, 가까운 곳·먼 곳, 서로 다른 기울기에서 약 20쌍을
   모읍니다. 두 카메라가 같은 호스트에 도착한 시각 차이 80 ms 이하의 최근
   프레임을 선택하는 소프트웨어 방식이며 하드웨어 트리거 동기화와 구별됩니다.
4. **G**로 계산해 회전·이동·재투영 RMS를 보고 **S**로 새 JSON에 저장합니다.
   출력은 `P_cam1 = R_2to1 @ P_cam2 + t_2to1_m`이며 단위는 m입니다.
   OpenCV가 반환하는 cam1→cam2 변환을 역변환하고 회전 직교성과 det=+1을
   검사합니다. 공장 내부 파라미터는 고정합니다. 보정은 왜곡을 제거한 픽셀에서
   최소제곱으로 두 뷰의 재투영 오차를 줄입니다. 쌍별 RMS가 2 px를 넘으면
   가장 큰 쌍부터 제외하며 최소 6쌍 및 전체의 70% 이상을 유지합니다.
5. **독립 검증.** 같은 설치에서 **보정에 쓰지 않은 자세**를 잡고 아래 명령을
   실행합니다. **F → 코너 번호 확인 → Enter**를 여러 거리와 기울기에서
   반복하고 **Q**를 누릅니다.

```bash
python stereo_board.py validate output/stereo.json --csv output/validation.csv
```

검증 CSV에는 두 카메라의 측정 시각, 유효 Depth 표본 수, 공통 XYZ, 축별
오차(cam2 측정 − cam1 측정), 3D 거리 오차, 보드 기울기, 별도 자세의
재투영 오차가 남습니다. 콘솔에는 유효점 수와 평균·최대·RMSE가 표시되고
유효점이 있으면 거리·기울기별 오차 그래프 PNG가 생성됩니다. 점 깊이는
정렬된 컬러 좌표의 5×5 패치에서 최소 5개 유효 표본의 중앙값으로 구합니다.
Depth 홀은 결측으로 남깁니다. 두 측정의 일치도를 실제 절대 위치에 대한
정확도로 해석하지 않습니다.

**객체 검출과 좌석 등록에서 재사용**

```bash
python main.py --serials 948522070995 148522071908 --stereo output/stereo.json \
  --jsonl output/objects_common.jsonl
python multiview.py calibrate output/seats_checked.npz \
  --serials 948522070995 148522071908 --seats 1 --stereo output/stereo.json
```

객체별 `position_3d_m`은 원래 카메라의 RGB 광학 좌표, `position_common_m`은
1번 RGB 광학 좌표입니다. 유효 Depth가 부족하면 좌표는 null입니다. 함께
기록되는 검출 신뢰도·유효 Depth 픽셀 수·시리얼·시각을 추적 모듈에 전달합니다.
좌석 등록에서는 각 A/B/C/D의 양방향 에피폴라 선 거리 중 큰 값이 5 px를
넘으면 대응점을 다시 선택합니다. 같은 선 위의 다른 점도 통과할 수 있어
물리적인 A/B/C/D 순서를 직접 확인해야 합니다. 카메라가 이동하면 새로
보정해야 하고, 시리얼·RGB 스트림 설정이 바뀌면 저장한 설정과 대조합니다.

`stereo_geometry.load_calibration(path)`로 결과를 읽은 뒤
`cal.get_camera_intrinsics(serial)`, `cal.pixel_to_3d(serial, (u,v), depth_m, timestamp)`,
`cal.transform_to_common_frame(serial, xyz_m)`를 호출할 수 있습니다.
`pixel_to_3d`는 유효 여부·시각·공통 XYZ를 반환하며 입력 Depth는 해당 RGB
픽셀에 정렬된 광학축 깊이(m)입니다. 표준 Brown 모델은 OpenCV로, 계수가 있는
inverse Brown 모델은 RealSense SDK 역투영으로 처리합니다. 변환을 지원하지
않는 모델은 이름과 함께 오류를 냅니다. [OpenCV 보정·3D 복원](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) · [RealSense 좌표계와 역투영](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

### 두 시점의 RGB 관측을 SeatSense 앱으로 보내기

**삼각대 없이 앱까지 시연**하려면 개발 PC에서 아래 명령을 실행합니다.

```bash
python demo_local.py
```

터미널에 뜬 로컬 관리자 주소를 열고 이번 시연용 아이디와 비밀번호로 로그인합니다.
배치도의 1번 좌석을 보면서 터미널에서 Enter를 누르면 두 시점의 모의 영상이 약
23초간 재생 처리됩니다. 빈자리, 한쪽에서 사람이 보임, 한쪽 상판만 변함, 두
상판의 같은 위치가 변함, 다시 비워짐 순서입니다. 사람 박스는 모의 입력이며
상판 변화 검출·두 시점 결합·앱 API·화면 실시간 갱신은 실제 코드를 실행합니다.
시연은 기존 Flask 앱을 임시 DB와 임시 비밀번호로 실행합니다. 끝난 뒤 Enter로
닫습니다. `--seat 138`처럼 시험 번호를 바꿀 수 있습니다. 앱 경로의 기본값은
`~/Desktop/hello_flask`이며 다른 PC에서는 `--app`과 `--flask-python`으로 지정합니다.

앱 없이 판정만 확인할 때는 `python multiview.py demo`를 실행합니다.
`--jsonl output/demo.jsonl`을 추가하면 카메라별 사람·소지품 후보, 상판 변화
영역, 두 시점의 판정이 기록됩니다. 각 실행에는 새 파일명을 사용합니다.

**카메라 고정 후 보정**할 때는 양쪽에서 보이는 좌석만 지정합니다. 아래 1번은
예시 번호입니다. 실제 좌석 번호를 여러 개 공백으로 구분해 넣을 수 있습니다.

```bash
python multiview.py calibrate output/reading_room.npz \
  --serials 948522070995 148522071908 --seats 1
python multiview.py live output/reading_room.npz \
  --jsonl "output/fusion_$(date +%Y%m%d_%H%M%S).jsonl"
```

1. 상판과 의자를 비우고 첫 카메라 창에서 **F**를 누릅니다. 빈 영상 15장의
   중앙값이 기준 영상이 됩니다.
2. 한 좌석이 사용하는 상판의 꼭짓점 네 개를 둘레를 따라 A, B, C, D 순으로
   누른 뒤 **Enter**를 누릅니다. **R**로 다시 선택할 수 있습니다. 이어서
   그 좌석에 앉은 사람의 몸통과 의자가 들어올 범위를 드래그하고 **Enter**를 누릅니다.
3. 두 번째 카메라에서도 **같은 실제 좌석과 같은 물리적 꼭짓점 A, B, C, D**를
   같은 순서로 선택합니다. 반대쪽에서 보면 A가 화면의 오른쪽에 있을 수 있습니다.
   보정은 시리얼 번호에 연결됩니다. 카메라 위치나 구도를 바꾸면 다시 보정합니다.
4. 먼저 `live` 창에서 좌석 윤곽과 카메라별 판정을 확인합니다. 실제 앱에 보내려면
   앱의 `/demo`에서 **카메라 API 모드**를 선택하고 아래처럼 서버 주소를 지정합니다.
   비밀번호는 앱에서 설정한 관리자 비밀번호입니다.

```bash
read -rsp 'SeatSense 관리자 비밀번호: ' SEAT_ADMIN_PASSWORD && echo
export SEAT_ADMIN_PASSWORD
python multiview.py live output/reading_room.npz --app-url http://127.0.0.1:5000
```

원격 앱에는 HTTPS 주소를 사용합니다. 앱은 `POST /api/action`의
`action=detect`, `source=camera`, `seat_id`, `occupied`, `belongings`를 받아
`/api/live`로 화면에 전달합니다. 소유자나 예약 변경 요청은 보내지 않습니다.

| 수업 기법 | 실제로 쓰이는 곳 |
| --- | --- |
| L01 배경 추출·영상 차분, L02 필터링·임계값·모폴로지 | 빈 상판과 비교하고 작은 변화 영역을 제거합니다. 기존 `DeskChangeDetector`를 양쪽에 재사용합니다. |
| L03 색 정보 | 밝기 차이와 채널 비율 변화를 함께 검사합니다. 밝기만 달라진 영역은 불확실할 수 있습니다. |
| L05 평면 호모그래피 | 카메라마다 상판 네 대응점으로 `getPerspectiveTransform`을 구하고 `warpPerspective`로 160×96 상판 영상에 정렬합니다. 사람과 의자의 위치는 각 카메라의 원본 RGB 영상에서 판단합니다. |
| 사전학습 객체 검출 | 전체 영상에서 사람과 등록된 소지품을 찾습니다. 중심점이 두 좌석에 겹치면 해당 좌석의 증거를 불확실로 둡니다. 의자 박스는 상판 가림의 단서로 사용합니다. |

두 카메라의 호스트 수신 시각 차이가 기본 0.5초 이내인 프레임을 한 번씩
짝지어 비교합니다. 한쪽에서만 사람이나 등록된 짐을 봐도 해당 관측을 사용할 수
있습니다. 미등록 물체는 양쪽 상판에서 변화 영역이 작은 쪽 면적의 25% 이상
겹칠 때 짐 후보가 됩니다. 어느 쪽에도 알려진 짐이 없고 두 상판 모두 변화가
없을 때만 짐 없음으로 처리합니다. 한쪽에만 생긴 차분, 가림, 미수신 등으로
사람·짐 중 하나라도 확정할 수 없으면 새 앱 상태를 보내지 않습니다.
기본값은 최소 세 쌍의 연속 관측과 사람·짐 있음 2초, 모두 없음 5초입니다.
현장 조정 전의 시작값입니다. 자동 반납과 짐의 소유자 결정은 별도 단계입니다.

### 단일 카메라 및 RGB-D 진단

현재 `config.json`은 USB 3.2로 연결된 RealSense 시리얼 `948522070995`
한 대를 640×480@30으로 사용하도록 설정되어 있습니다.

연결된 RealSense 목록만 확인할 때는 YOLO를 로드하지 않습니다.

```bash
python main.py --list-cameras
```

RealSense RGB-D 입력만 100프레임 진단할 때도 YOLO를 로드하지 않습니다.

```bash
python main.py --camera-only --serial 948522070995 \
  --no-display --max-frames 100
```

초기 자동 노출 및 Depth 안정화를 위해 기본 30프레임은 측정에서 제외합니다.
진단 결과에는 수신 프레임, 누락 프레임, timeout, 측정 FPS, 유효 Depth 비율,
Color/Depth timestamp 차이와 timestamp 역행 횟수가 표시됩니다.

두 대를 동시에 진단하려면 각 카메라가 단독 테스트를 통과한 뒤 serial을
명시합니다. 카메라마다 독립 pipeline과 크기 1의 최신 프레임 큐를 사용합니다.
미리보기는 `camera_1`, `camera_2` 별도 창으로 생성되어 좌우로 배치됩니다.
각 창의 FPS는 해당 카메라가 실제로 처리된 FPS이며, JSON의 `pipeline_fps`에는
두 카메라를 합친 전체 처리 FPS가 별도로 기록됩니다.

```bash
python main.py --camera-only \
  --serials 948522070995 148522071908 \
  --no-display --max-frames 100
```

```bash
python main.py
```

시리얼을 명시해서 실행하려면:

```bash
python main.py --input-type realsense --serial 948522070995
```

일반 웹캠 0번:

```bash
python main.py --input-type video --source 0
```

RGB 동영상 파일:

```bash
python main.py --input-type video --source path/to/video.mp4
```

화면 없이 10프레임만 처리하고 팀원 연동용 JSONL을 저장하는 스모크 테스트:

```bash
python main.py --input-type video --source path/to/video.mp4 \
  --no-display --max-frames 10 \
  --jsonl output/detections.jsonl
```

미리보기에서 `q`를 누르면 정상 종료합니다. 모델, confidence, 입력 장치와
대상 클래스는 `config.json`에서 바꿀 수 있고 주요 실행 옵션은 CLI에서
덮어쓸 수 있습니다.

## COCO 의자 검출

고정한 카메라의 전체 RGB 영상에서 의자를 찾습니다. 영역 지정 없이 실행하면
검출된 의자마다 하늘색 박스와 `chair` 신뢰도가 표시됩니다.

```bash
python main.py --chair-only --serial 948522070995 \
  --jsonl "output/chair_coco_$(date +%Y%m%d_%H%M%S).jsonl"
```

빈 의자를 책상 안쪽에 넣고, 반쯤 빼고, 옆으로 돌린 뒤 각 위치에서 잠시 멈춰
같은 의자에 박스가 뜨는지 확인합니다. 마지막에 가방을 올려 가려진 상태도
비교할 수 있습니다. `q`로 종료하면 검출이 없었던 프레임까지 JSONL에 남습니다.
각 프레임의 의자 위치와 신뢰도를 기록하며, 의자 위 소지품 판정은 별도 실험입니다.

## 책상 한 구역에서 미등록 물체 실험

책상 상판을 비우고 실행하면 실시간으로 구도를 맞출 수 있습니다. 이 실험은 RealSense 컬러
스트림으로 선택한 구역을 촬영하고, 빈 상판과의 RGB 변화로 물체 후보를 표시합니다.

```bash
session="output/library_$(date +%Y%m%d_%H%M%S)"
python desk.py record "$session" --serial 948522070995
python desk.py replay "$session"
```

1. `LIVE` 창을 보며 카메라를 움직입니다. 원하는 구도에 카메라를 고정한 뒤
   창에서 **F**를 누르면 그 순간의 화면이 멈춥니다.
2. 멈춘 화면에서 빈 상판을 드래그하고 **Enter**를 누릅니다. 선택 중 **c**를 누르면
   실시간 화면으로 돌아갑니다. 확정 후에는 기준 영상을 준비하는 약 3초간 책상을 비워 둡니다.
3. `READY`에서 **R**을 누르면 저장이 시작됩니다. 구도를 다시 잡으려면 **B**를 누릅니다.
   `REC`가 뜨면 천을 놓고 손을 뗍니다.
   10초 정도 그대로 뒀다가 치워 보세요. 그림자만 드리우는 구간도 비교해 보세요.
4. 영상 창에서 **Q**를 누르면 종료합니다. 기본은 초당 약 5프레임, 최대 150프레임입니다.
   다음 촬영에는 새 폴더 이름을 사용합니다.
5. `replay`는 저장된 기준과 관측 프레임을 순서대로 읽어 같은 검사를 수행합니다.
   이전 RGB-D 녹화도 동일한 명령으로 컬러 영상만 재생·판정합니다.
   터미널로만 확인하려면 `--no-display`를 붙입니다.

키 입력을 받을 GUI가 없는 환경에서는 `record --no-display`에 `--roi X Y W H`를 함께
지정하면 기준 수집 후 자동으로 녹화합니다.

`reference.npz`는 15개 표본의 픽셀별 중앙값과 serial, 원본 해상도, ROI 좌표를
보관합니다. 각 `frame_*.npz`에는 `color_bgr`(uint8), 프레임 번호,
컬러 타임스탬프(ms)가 들어갑니다. 좌표는
ROI 왼쪽 위를 원점으로 사용합니다. 카메라 위치나 각도가 달라지면 새 기준을 촬영합니다.

빈 상판과 비교해 RGB 채널 값과 채널 간 색 비율이 함께 변한 영역을 구합니다.
같은 위치에서 3개 표본 연속 변화가 나타나고 열림·닫힘 연산
후 연결된 영역이 ROI의 1% 이상(최소 12픽셀)이면 `object_candidate`로 표시합니다.
`uncertain`은 밝기만 변하거나 색 변화가 아직 충분히 지속되지 않은 경우 등이며,
`no_change`는 유의한 변화가 관측되지 않은 경우입니다. 머무르는 손도 물체 후보가
될 수 있습니다. 이 실험의 출력은 상판 관측용이며 기존 YOLO 좌석 JSON과는 독립적입니다.

## 출력 인터페이스

팀별 입력·출력, 두 팀원에게 전달할 프롬프트와 제안서의 평가 기준은
[팀 작업 분담](TEAM_HANDOFF.md)에 정리했습니다.

`VisionDetector.process_frame(...)`는 JSON 직렬화가 가능한 `dict`를
반환합니다. 각 detection에는 class, confidence, bbox, center, `depth_m`,
`position_3d_m`, 유효 depth 픽셀 수와 좌석 후보가 들어갑니다. RGB 영상처럼
depth가 없는 입력은 거리와 3D 위치를 `null`로 둡니다. 임의의 Tracking ID는
만들지 않습니다.

객체 거리는 bbox 중앙 50% 영역에서 0.1~10m 사이의 유효 픽셀 median으로
계산합니다. 유효 픽셀이 20개보다 적거나 0/포화값뿐이면 임의 거리로
대체하지 않고 `null`을 반환합니다. `position_3d_m`은 해당 카메라의 좌표이고,
`main.py --stereo`는 보정 파일을 적용한 `position_common_m`도 제공합니다.

`seat_rois.json`은 12석 편집용 템플릿입니다. 현재 좌표는 실제 현장 보정값이
아니므로 `calibrated`가 `false`이며 좌석 후보를 만들지 않습니다. 실제 화면에
맞춰 rectangle/polygon 좌표를 수정한 뒤 `calibrated`를 `true`로 바꾸고 다음과
같이 실행합니다.

```bash
python main.py --seat-rois seat_rois.json
```

COCO 사전학습 모델은 `person`, `chair`, `backpack`, `handbag`, `suitcase`, `laptop`,
`cell phone`, `book`을 지원하며 현재 검출 대상으로 설정했습니다. `book`은
COCO class 73이지만 작은 객체라 실제 도서관 환경에서 성능 검증이 필요합니다.

## 파일 역할

- `main.py`: CLI 실행과 화면/JSONL 출력
- `camera.py`: 웹캠, RGB 동영상, 단일/다중 RealSense RGB-D 입력
- `depth.py`: bbox 중앙 영역 median depth와 3D 좌표 계산
- `detector.py`: YOLO 추론과 검출 결과 데이터 구성
- `seat_roi.py`: ROI 겹침 후보와 좌석별 기본 관측 생성
- `config.json`: 모델 및 검출 설정
- `seat_rois.json`: 12개 좌석 ROI 편집 템플릿
- `requirements.txt`: 최소 실행 의존성

## 현재 확인한 환경

- Python 3.13.13 / OpenCV 5.0.0 / NumPy 2.5.1
- PyTorch 2.15.0+cpu / Ultralytics 8.4.215
- 웹캠 0번에서 실제 1프레임 입력 및 YOLO 추론 정상
- 샘플 RGB 영상 2프레임 처리, 사람 검출 및 JSONL 저장 정상
- USB 3.2 RealSense `948522070995`의 정렬된 RGB-D 640×480@30 수신 정상
- 실제 카메라 RGB-D+YOLO 통합 10프레임 및 JSONL 출력 정상(CPU 평균 약 12 FPS)
- 현재 테스트 화면에는 지원 객체가 없어 객체별 실제 depth 표본은 미확인
- 모의 depth로 거리, 3D 위치와 ROI 후보 출력 정상
- 미리보기 창과 `q` 종료는 자동 테스트하지 않음
- 2026-11-20 USB 3.2 RealSense 2대 각각 warm-up 30 + 측정 100프레임 정상
- 단일 진단에서 양쪽 모두 약 30 FPS, 누락/timeout/timestamp 역행 0
- 듀얼 RGB-D 동시 진단에서 양쪽 100프레임, queue drop 0
- 듀얼 YOLO26 CPU 경로에서 카메라별 serial/frame ID 분리 및 JSONL 출력 정상
- 듀얼 YOLO26 CPU 40프레임 테스트에서 queue drop은 camera_1 51,
  camera_2 52로 측정됨. 합산 60 FPS 입력보다 CPU 추론이 느린 것이 원인이며,
  최신 프레임을 유지하고 오래된 프레임을 버리는 동작은 정상
