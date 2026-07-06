import numpy as np
import matplotlib.pyplot as plt

class MVNgraph():
    def __init__(self):
        self.points = {}
        self.scale = 100

    def parabole(self, x1, y1, x2, y2, x3, y3):
        A = np.array([[x1**2, x1, 1], 
                      [x2**2, x2, 1], 
                      [x3**2, x3, 1]])
        b_ = np.array([[y1],
                      [y2],
                      [y3]])
        a, b, c = np.linalg.solve(A,b_)
        parlist = []
        for x in np.linspace(x1, x3, 100):
            y = a*x**2 + b*x + c
            parlist.append((x, y))
        return [parlist, x1, y1, x2, y2, x3, y3]

    def interpolate_y(self, xs, ys, x_query):
        for i in range(len(xs) - 1):
            if xs[i] <= x_query <= xs[i + 1]:
                return ys[i] + (ys[i + 1] - ys[i]) * ((x_query - xs[i]) / (xs[i + 1] - xs[i]))
            
    def rotate(self, beam, x, y):
        a = np.radians(beam.angle)

        return (beam.x1 + x*np.cos(a) - y*np.sin(a),
                beam.y1 + x*np.sin(a) + y*np.cos(a))

    def add_beam(self, beam, *pointtuples):
        if beam not in self.points:
            self.points[beam] = []
        
        pointlist = []
        labels = []

        for arg in pointtuples:
            # if it is a parabolic segment
            if isinstance(arg, list): 
                pointlist.extend(arg[0])
                # use only the characteristic points for the labels
                labels.append((str(abs(arg[2])), arg[1], arg[2]))
                labels.append((str(abs(arg[4])), arg[3], arg[4]))
                labels.append((str(abs(arg[6])), arg[5], arg[6]))

            # if not a parabolic segment, so just a point
            else:
                pointlist.append(arg)
                labels.append((str(abs(arg[1])), arg[0], arg[1]))

        x, y = zip(*pointlist)
        x = list(x)
        y = list(y)
        y_max = max(y, key=abs)
        if y_max == 0:
            y_max = beam.length/3
        self.scale = min(self.scale, abs(beam.length/(2*y_max)))
        self.points[beam] = {"x": x, "y": y, "labels": labels, "flip_sign": False,}

    def flip_sign(self, beam):
        self.points[beam]["flip_sign"] = True

    def find_crossings(self, x, y):
        crossings = []

        for i in range(len(x)-1):
            x1, x2 = x[i], x[i+1]
            y1, y2 = y[i], y[i+1]

            if y1 == 0:
                crossings.append(x1)
            elif y1 * y2 < 0:
                crossings.append(x1 + (-y1)*(x2-x1)/(y2-y1))

        if y[-1] == 0:
            crossings.append(x[-1])

        return crossings
    
    def plot(self):
        axs = plt.gca()
        axs.axis("equal")

        for beam in self.points:
            self.plot_beam(axs, beam)

    def plot_beam(self, axs, beam):
        data = self.points[beam]
        x = data["x"]
        y = data["y"]
        labels = data["labels"]

        # draw the line through the data points
        y_scaled = [yi * self.scale for yi in y]
        x_rot, y_rot = zip(*(self.rotate(beam, xi, yi)
                     for xi, yi in zip(x, y_scaled)))
        axs.plot(x_rot, y_rot, color="red")

        # draw the beam itself
        x0, y0 = self.rotate(beam, x[0], 0)
        x1, y1 = self.rotate(beam, x[-1], 0)
        axs.plot([x0, x1], [y0, y1], color="black")

        # find places where line crosses zero or is zero
        crossings = self.find_crossings(x, y)

        if crossings[0] != x[0]:
            crossings.insert(0, x[0])

        if crossings[-1] != x[-1]:
            crossings.append(x[-1])

        # draw the signs, depending on which type of graph
        self.draw_signs(beam, crossings, x, y)

        # draw labels with values at characteristic points
        self.plot_labels(axs, beam, labels)


    def plot_labels(self, axs, beam, labels):
        # plot labels at characteristic points
        value, x_label, y_label = zip(*labels)
        y_label = [y_label_i*self.scale for y_label_i in y_label]
        x_label_rotated, y_label_rotated = zip(*(self.rotate(beam, xi, yi)
                                        for xi, yi in zip(x_label, y_label)))
        for i in range(len(x_label_rotated)):
            axs.annotate(value[i], (x_label_rotated[i], y_label_rotated[i]), ha='center',va='center')


class Mgraph(MVNgraph):
    def draw_signs(self, beam, crossings, x, y):
        y = [yi*self.scale for yi in y]
        # determine position for sign and plot it
        for i in range(len(crossings) - 1):
            c1 = crossings[i]
            c2 = crossings[i+1]
            x_mid = (c1 + c2) / 2
            y_mid = self.interpolate_y(x, y, x_mid)

            theta = np.linspace(0, np.pi, 100)
            r = min((y_mid/3, 0.6*(c2-c1)/2 * y_mid/abs(y_mid)), key=abs) # make sure radius not too big
            x_sign = x_mid + r * np.cos(theta)
            y_sign = y_mid/3 + r * np.sin(theta) 
            x_sign_rotated, y_sign_rotated = zip(*(self.rotate(beam, xi, yi)
                                            for xi, yi in zip(x_sign, y_sign)))
            plt.plot(x_sign_rotated, y_sign_rotated, linewidth=1.5, color='black')

                   
class Vgraph(MVNgraph):
    def draw_signs(self, beam, crossings, x, y):
        y = [yi*self.scale for yi in y]
        # draw shear symbol
        flip_sign = self.points[beam]["flip_sign"]
        for i in range(len(crossings) - 1):
            c1 = crossings[i]
            c2 = crossings[i+1]
            x_mid = (c1 + c2) / 2
            y_mid = self.interpolate_y(x, y, x_mid)
            du = y_mid / -5 if flip_sign else y_mid / 5
            x_sign = [x_mid - du, x_mid, x_mid, x_mid + du]
            y_sign = [y_mid/3 - abs(du)/2, y_mid/3 - abs(du)/2, y_mid/3 + abs(du)/2, y_mid/3 + abs(du)/2] 
            x_sign_rotated, y_sign_rotated = zip(*(self.rotate(beam, xi, yi)
                                            for xi, yi in zip(x_sign, y_sign)))
            plt.plot(x_sign_rotated, y_sign_rotated, linewidth=1.5, color='black')



class Ngraph(MVNgraph):
    def draw_signs(self, beam, crossings, x, y):
        y = [yi*self.scale for yi in y]
        # draw +/- symbol
        # determine position for sign and plot it
        flip_sign = self.points[beam]["flip_sign"]
        for i in range(len(crossings) - 1):
            c1 = crossings[i]
            c2 = crossings[i+1]
            x_mid = (c1 + c2) / 2
            y_mid = self.interpolate_y(x, y, x_mid)
            du = abs(y_mid / 4)

            # draw minus sign
            x_sign = [x_mid - du/2, x_mid + du/2]
            y_sign = [y_mid/3, y_mid/3] 
            x_sign_rotated, y_sign_rotated = zip(*(self.rotate(beam, xi, yi)
                                            for xi, yi in zip(x_sign, y_sign)))
            plt.plot(x_sign_rotated, y_sign_rotated, linewidth=1.5, color='black')

            # draw plus sign (add a vertical line) 
            if (y_mid > 0 and not flip_sign) or (y_mid < 0 and flip_sign):
                x_sign = [x_mid, x_mid]
                y_sign = [y_mid/3 - du/2, y_mid/3 + du/2] 
                x_sign_rotated, y_sign_rotated = zip(*(self.rotate(beam, xi, yi)
                                            for xi, yi in zip(x_sign, y_sign)))
                plt.plot(x_sign_rotated, y_sign_rotated, linewidth=1.5, color='black')
    