import os
import sys
import argparse
import time
import cv2
from collections import OrderedDict

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QDialog
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QTableWidget, QLineEdit, QLabel, QPushButton, QTableWidgetItem
from PySide6.QtGui import QAction, QImage
from PySide6.QtCore import Qt, QSize, Slot, QThread

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from utils.config import _C as cfg, update_config
from utils.logger import init_logger, get_logger
from obj_detectors.obj_detector import ObjectDetector
from utils.medialoader import MediaLoader


ROW = OrderedDict({"chicken": "닭고기류",
                   "sauce1": "소스 1",
                   "sauce2": "소스 2",
                   "powder": "파우더",
                   "quantity": "발주물량",
                   "match": "집계물량"})
PROD = [
    {
        "name": "수원왕갈비 꾸닭",
        "chicken": ["태음융융소금염지닭", 50],
        "sauce1": ["닭갈비간장", 25],
        "sauce2": ["갈비레이", 25],
        "powder": ["에어크런치", 50],
        "quantity": 50,
    },
    {
        "name": "오리지널 치밥",
        "chicken": ["후라이드염지닭", 30],
        "sauce1": ["치밥버무림소스", 15],
        "sauce2": ["치밥벌크소스", 15],
        "powder": ["크리스피파우더", 30],
        "quantity": 30,
    },
    {
        "name": "소이퐁 튀닭",
        "chicken": ["후라이드염지닭", 50],
        "sauce1": ["소이퐁소스", 50],
        "sauce2": [],
        "powder": ["우리쌀 후레이크파우더", 50],
        "quantity": 50,
    },
    {
        "name": "공주매콤 닭갈비",
        "chicken": ["태음융융소금염지닭", 50],
        "sauce1": ["닭갈비 간장양념", 50],
        "sauce2": [],
        "powder": [],
        "quantity": 50,
    },
    {
        "name": "크리스피 튀닭",
        "quantity": 50,
        "chicken": ["핫커리염지닭", 50],
        "sauce1": ["맛있게매운소스", 25],
        "sauce2": ["프리마늘소스", 25],
        "powder": ["크리스피파우더", 50]
    },
]


class AnalysisThread(QThread):
    def __init__(self, parent=None, medialoader=None, statusbar=None, detector=None):
        super().__init__(parent=parent)
        self.medialoader = medialoader
        self.sbar = statusbar
        self.detector = detector
        self.stop = False

    def run(self):
        self.medialoader.restart()
        self.sbar.showMessage("입력 영상 송출")
        self.stop = False
        while True:
            frame = self.medialoader.get_frame()
            if frame is None or self.stop is True:
                break

            im = self.detector.preprocess(frame)
            _pred = self.detector.detect(im)
            _pred, _det = self.detector.postprocess(_pred)


            # cv2.imshow("input", frame)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
            # img.save("/home/dongle94/t.jpg")

            time.sleep(0.005)

        # cv2.destroyAllWindows()
        self.sbar.showMessage("입력 영상 종료")


class ProdTable(QTableWidget):
    def __init__(self, col, row):
        super().__init__()
        self.col = col
        self.row = row


    def set_header(self):
        # Set vertical header
        self.setRowCount(1 + len(self.row))
        self.setVerticalHeaderLabels([""] + list(self.row.values()))

        # Set Horizontal header
        self.horizontalHeader().setVisible(False)
        self.setColumnCount(len(self.col) * 3)
        for i, col in enumerate(self.col):
            self.setSpan(0, 3 * i, 1, 3)
            self.setItem(0, 3 * i, QTableWidgetItem(col["name"]))
            self.item(0, 3 * i).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # for _i in range(3):
            #     self.horizontalHeader().setSectionResizeMode(i * 3 + _i,  )

        # Set span not prod
        for i in range(len(self.col)):
            self.setSpan(5, 3 * i, 1, 3)
            self.setSpan(6, 3 * i, 1, 3)

    def set_contents(self):
        for c, col in enumerate(self.col):
            for r, (k, v) in enumerate(list(self.row.items())[:5]):
                if k == "quantity":
                    self.setItem(r + 1, 3 * c, QTableWidgetItem(str(col[k])))
                    self.item(r + 1, 3 * c).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    continue

                if col[k]:
                    self.setItem(r + 1, 3 * c, QTableWidgetItem(col[k][0]))
                    self.setItem(r + 1, 3 * c + 1, QTableWidgetItem(str(col[k][1])))
                    self.setItem(r + 1, 3 * c + 2, QTableWidgetItem(str(0)))
                else:
                    self.setItem(r + 1, 3 * c, QTableWidgetItem("-"))

        self.resizeColumnsToContents()

    def show_table(self):
        self.set_header()
        self.set_contents()

    def clear_table(self):
        self.clear()

        # Delete each rows
        r_cnt = self.rowCount()
        for r in range(r_cnt):
            self.removeRow(r_cnt - r - 1)


class MainWidget(QWidget):
    def __init__(self, parent):
        super().__init__()

        # init MainWidget
        self.logger = get_logger()
        self.logger.info("Create Main QWidget - start")
        self.main_window = parent

        # top - layout
        self.top = QHBoxLayout()
        self.el_date = QLineEdit()
        self.el_client = QLineEdit()
        self.el_prod_name = QLineEdit()
        self.bt_search = QPushButton("발주 조회")
        self.bt_reset = QPushButton("초기화")

        self.top.addWidget(QLabel("날짜:"))
        self.top.addWidget(self.el_date)
        self.top.addWidget(QLabel("발주사:"))
        self.top.addWidget(self.el_client)
        self.top.addWidget(QLabel("제품명:"))
        self.top.addWidget(self.el_prod_name)
        self.top.addWidget(self.bt_search)
        self.top.addWidget(self.bt_reset)

        # middle - table
        self.middle = ProdTable(col=PROD, row=ROW)

        # bottom - layout
        self.bottom = QHBoxLayout()
        self.bt_start = QPushButton("수량 학인 시작")
        self.bt_stop = QPushButton("수량 확인 종료")
        # self.bt_show = QPushButton("실시간 영상 보기")
        self.bt_start.setDisabled(True)
        self.bt_stop.setDisabled(True)
        # self.bt_show.setDisabled(True)

        self.bottom.addWidget(self.bt_start)
        self.bottom.addWidget(self.bt_stop)
        # self.bottom.addWidget(self.bt_show)

        # Main Widget
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addLayout(self.top)
        self.main_layout.addWidget(self.middle)
        self.main_layout.addLayout(self.bottom)

        # Status bar
        self.sbar = self.main_window.statusBar()

        # Set Input Loader
        self.medialoader = self.main_window.media_loader
        self.medialoader.start()
        self.medialoader.pause()

        self.analysis_thread = AnalysisThread(self, self.medialoader, self.sbar, self.main_window.obj_detector)

        # signals & slots
        self.bt_search.clicked.connect(self.set_table)
        self.bt_reset.clicked.connect(self.clear_table)
        self.bt_start.clicked.connect(self.start_analysis)
        self.bt_stop.clicked.connect(self.stop_analysis)
        # self.bt_show.clicked.connect(self.show_analysis)

        self.sbar.showMessage("프로그램 준비 완료")
        self.logger.info("Create Main QWidget - end")


    @Slot()
    def set_table(self):
        self.logger.info("set_table - start")
        self.middle.show_table()

        self.bt_start.setDisabled(False)
        self.bt_stop.setDisabled(False)
        # self.bt_show.setDisabled(False)
        self.sbar.showMessage("발주 내용 조회")
        self.logger.info("set_table - end")

    @Slot()
    def clear_table(self):
        self.logger.info("clear_table - start")
        self.middle.clear_table()

        self.bt_start.setDisabled(True)
        self.bt_stop.setDisabled(True)
        # self.bt_show.setDisabled(True)
        self.sbar.showMessage("테이블 초기화")
        self.logger.info("clear_table - end")

    @Slot()
    def start_analysis(self):
        self.logger.info("Start analysis")
        self.analysis_thread.start()

    @Slot()
    def stop_analysis(self):
        self.logger.info("Stop analysis")
        self.analysis_thread.stop = True
        self.analysis_thread.exit()


class DaolCND(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        # init gui
        self.logger = get_logger()
        self.logger.info("Init DaolCND QMainWindow - start")

        # resize window
        self.geo = self.screen().availableGeometry()
        self.resize(QSize(self.geo.width() * 0.5, self.geo.height() * 0.5))

        # MenuBar
        self.mbar = self.menuBar()
        self.file_menu = self.mbar.addMenu("File")

        # Menu - Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.statusBar().showMessage("프로그램 준비 시작", 0)

        # Get Object Detector
        self.config = config
        self.obj_detector = ObjectDetector(cfg=self.config)
        self.media_loader = MediaLoader(source="0", logger=self.logger)

        # Set Central Widget
        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)

        self.logger.info("Init DaolCND QMainWindow - end")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='configuration')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Get Configuration
    args = args_parse()
    _cfg = args.config
    update_config(cfg, _cfg)

    # initialize logger
    init_logger(cfg=cfg)

    app_gui = DaolCND(cfg)
    app_gui.setWindowTitle("다올씨앤디")
    app_gui.show()

    sys.exit(app.exec())
