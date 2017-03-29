from lane_data_types import Line, Lane, LaneHistory

def test_declare_line():
    theLine = Line()
    print('test_declare_line: ok')

def test_declare_lane():
    theLane = Lane()
    print('test_declare_lane: ok')

def test_declare_laneHistory():
    theLaneHist = LaneHistory()
    print('test_declare_laneHistory: ok')


def main():
    test_declare_line()
    test_declare_lane()
    test_declare_laneHistory()
main()
