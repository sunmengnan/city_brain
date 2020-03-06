from operator import itemgetter
import numpy
import matplotlib
import matplotlib.pyplot
import numpy as np


class Simple_polygon:

    def __init__(self, array):
        self.array = array

    def close_line_to_polygon(self):
        a = self.array[0]
        b = self.array[len(self.array) - 1]
        if (a == b).all():
            pass
        else:
            self.array.append(a)

    def find_leftmost_point(self):
        leftmost_point = None
        leftmost_x = None
        for point in self.array:
            x = point[0]
            if leftmost_x == None or x < leftmost_x:
                leftmost_x = x
                leftmost_point = point
        return leftmost_point

    def find_rightmost_point(self):
        rightmost_point = None
        rightmost_x = None
        for point in self.array:
            x = point[0]
            if rightmost_x == None or x > rightmost_x:
                rightmost_x = x
                rightmost_point = point
        return rightmost_point

    def is_point_above_the_line(self, point, line_points):
        """return 1 if point is above the line
           return -1 if point is below the line
           return  0 if point is lays on the line"""
        px, py = point
        P1, P2 = line_points
        P1x, P1y = P1[0], P1[1]
        P2x, P2y = P2[0], P2[1]
        array = numpy.array([
            [P1x - px, P1y - py],
            [P2x - px, P2y - py],
        ])
        det = numpy.linalg.det(array)
        sign = numpy.sign(det)
        return sign

    def sort_array_into_A_B_C(self, line_points):
        A_array, B_array, C_array = [], [], []
        for point in self.array:
            x, y = point
            sing = self.is_point_above_the_line((x, y), line_points)
            if sing == 0:
                C_array.append(point)
            elif sing == -1:
                A_array.append(point)
            elif sing == 1:
                B_array.append(point)
        return A_array, B_array, C_array

    def sort_and_merge_A_B_C_arrays(self, A_array, B_array, C_array):
        A_C_array = [*A_array, *C_array]
        A_C_array.sort(key=itemgetter(0))
        B_array.sort(key=itemgetter(0), reverse=True)
        merged_arrays = [*A_C_array, *B_array]
        self.array = merged_arrays

    def show_image(self, array, line_points, A_array, B_array, C_array):
        [(x_lm, y_lm), (x_rm, y_rm)] = line_points
        x = [x[0] for x in array]
        y = [y[1] for y in array]
        Ax = [x[0] for x in A_array]
        Ay = [y[1] for y in A_array]
        Bx = [x[0] for x in B_array]
        By = [y[1] for y in B_array]
        Cx = [x[0] for x in C_array]
        Cy = [y[1] for y in C_array]
        matplotlib.pyplot.plot(Ax, Ay, 'o', c='orange')  # below the line
        matplotlib.pyplot.plot(Bx, By, 'o', c='blue')  # above the line
        matplotlib.pyplot.plot(Cx, Cy, 'o', c='black')  # on the line
        matplotlib.pyplot.plot(x_lm, y_lm, 'o', c='green')  # leftmost point
        matplotlib.pyplot.plot(x_rm, y_rm, 'o', c='red')  # rightmost point
        x_plot = matplotlib.pyplot.plot([x_lm, x_rm], [y_lm, y_rm], linestyle=':', color='black',
                                        linewidth=0.5)  # polygon's division line
        x_plot = matplotlib.pyplot.plot(x, y, color='black',
                                        linewidth=1)  # connect points by line in order of apperiance
        matplotlib.pyplot.show()

    def main(self, plot=False):
        print(self.array)
        x_lm, y_lm = self.find_leftmost_point()
        x_rm, y_rm = self.find_rightmost_point()
        line_points = [(x_lm, y_lm), (x_rm, y_rm)]

        A_array, B_array, C_array = self.sort_array_into_A_B_C(line_points)
        self.sort_and_merge_A_B_C_arrays(A_array, B_array, C_array)
        self.close_line_to_polygon()
        if plot:
            self.show_image(self.array, line_points, A_array, B_array, C_array)
        self.array = self.array[:-1]
        return self.array


if __name__ == "__main__":
    array = [
        np.array([120.180717, 30.200412]),
        np.array([120.181164, 30.200398]),
        np.array([120.180813, 30.200255]),
        np.array([120.181080, 30.200614]),
    ]
    spg = Simple_polygon(array)
    polygon_array = spg.main(plot=True)
