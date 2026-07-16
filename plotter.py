from math import sqrt, degrees
import numpy as np
import matplotlib.pyplot as plt
import sympy
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
})
from matplotlib.patches import Arc, FancyArrowPatch, Polygon, Circle, FancyArrow
import matplotlib.ticker as plticker
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import hashlib
from sympy.functions.elementary.trigonometric import atan2
from scipy.ndimage import rotate


FORCE_COLORS = {
    'red': '#800000',
    'green': '#00CC99',
    'blue': '#31859B',
}

class Point():
    """Point class represents a point in 2D space with optional label and display properties.
    """
    def __init__(self, x: float, y:float ,label: str='', labelpos:tuple[str,str]=('top', 'center'),z:float = 0, is_opaque: bool = False, is_dashed: bool = False) -> None:
        """ function to create a point in 2D space with optional label and display properties.

        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.
            label (str, optional): Label for the point. Defaults to ''.
            labelpos (tuple[str,str], optional): Position of the label. Defaults to ('top', 'center').
            z (float, optional): z-coordinate of the point. Defaults to 0.
            is_opaque (bool, optional): Whether the point is opaque. Defaults to False.
            is_dashed (bool, optional): Whether the point is dashed. Defaults to False.
        """
        self.x = x
        self.y = y
        self.z = z
        self.label = label
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.is_opaque = is_opaque
        self.is_dashed = is_dashed

    def __repr__(self) -> str:
        return f'Point {self.label}: x={self.x}, y={self.y}, z={self.z}'
    
    def angle(self, point: "Point") -> float:
        """Returns the angle in degrees between the line connecting this point to another point and the positive x-axis."""
        return degrees(atan2(self.y - point.y, self.x - point.x))
    
    def distance_to(self, point: "Point") -> float:
        """Returns the Euclidean distance between this point and another point."""
        return sqrt((self.x - point.x)**2 + (self.y - point.y)**2)
        
    def make_opaque(self) -> None:
        self.is_opaque = True

    def make_dashed(self) -> None:
        self.is_dashed = True

class Beam():
    """Beam class represents a beam in 2D space defined by its start and end points, with optional label and display properties."""
    def __init__(self, begin: Point, end: Point, label:str=None, labelpos:tuple[str,str]=('top', 'center'), anglelabel:bool=False, anglelabelflip:bool=False, is_opaque:bool=False, is_dashed:bool=False) -> None:
        self.begin = begin
        self.x1 = begin.x
        self.y1 = begin.y
        self.end = end
        self.x2 = end.x
        self.y2 = end.y
        self.label = label
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.anglelabel =anglelabel
        self.labelflip = anglelabelflip
        self.is_opaque = is_opaque
        self.is_dashed = is_dashed
        self.hinges = []
        
    @property
    def length(self) -> float:
        """Returns the length of the beam."""
        return sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def angle(self) -> float:
        """Returns the angle of the beam in degrees."""
        return degrees(atan2(self.y2 - self.y1, self.x2 - self.x1))  
    
    def add_hinge(self, loc: str = 'start') -> None:
        if loc == 'start':
            self.hinges.append((self.begin, True))
        else:
            self.hinges.append((self.end, False))

    def make_opaque(self) -> None:
        self.is_opaque = True

    def make_dashed(self) -> None:
        self.is_dashed = True

class ParabolicBeam(Beam):
    """ParabolicBeam class represents a parabolic beam in 2D space defined by its start and end points, with optional label and display properties.
    The parabolic shape is defined by the mid_deflection parameter, which specifies the maximum deflection of the beam at its midpoint."""
    def __init__(self, begin: Point, mid_deflection: float, end: Point, label:str=None, labelpos:tuple[str,str]=('top', 'center'), anglelabel:bool=False, anglelabelflip:bool=False, is_opaque:bool=False) -> None:
        super().__init__(begin, end, label, labelpos, anglelabel, anglelabelflip, is_opaque)
        self.ym = mid_deflection

class DeformedBeam(Beam):
    """DeformedBeam class represents a deformed beam in 2D space defined by its start and end points, with optional label and display properties."""
    def __init__(self, begin: Point, end: Point, variable=sympy.symbols('x'), expression=0, label:str=None, labelpos:tuple[str,str]=('top', 'center'), anglelabel:bool=False, anglelabelflip:bool=False, is_opaque:bool=False) -> None:
        super().__init__(begin, end, label, labelpos, anglelabel, anglelabelflip, is_opaque)
        self.function = sympy.lambdify(variable, expression)

class Support():
    """Support class represents a support in 2D space defined by its location and type."""
    def __init__(self, point: Point, support_type: str = 'fixed', angle: float = 0.0, is_opaque: bool = False) -> None:
        self.loc = point
        self.x = point.x
        self.y = point.y
        self.set_type(support_type)
        self.angle = angle
        self.is_opaque = is_opaque

    def set_type(self, support_type) -> None:
        if support_type not in ['fixed', 'roller', 'pinned']:
            raise ValueError("Use either 'pinned', 'roller' or 'fixed' as support type")
        else:
            self._type = support_type

    def make_opaque(self) -> None:
        self.is_opaque = True

class RotationSpring():
    """RotationSpring class represents a rotational spring in 2D space defined by its location and properties."""
    def __init__(self, point: Point, value: float=None, unit:str='kNm/rad', labelpos:tuple[str,str]=('top', 'center'),alternative_label:str='', is_opaque: bool = False) -> None:
        self.x = point.x
        self.y = point.y
        self.value = value
        self.unit = unit
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.alt_label = alternative_label
        self.is_opaque = is_opaque

    def make_opaque(self) -> None:
        self.is_opaque = True

class TranslationSpring():
    """TranslationSpring class represents a translational spring in 2D space defined by its location and properties."""
    def __init__(self, startpoint: Point, endpoint: Point, value: float=None, unit:str='kN/m', labelpos:tuple[str,str]=('top', 'center'),alternative_label:str='', is_opaque: bool = False) -> None:
        self.x1 = startpoint.x
        self.y1 = startpoint.y
        self.x2 = endpoint.x
        self.y2 = endpoint.y
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.value = value
        self.unit = unit
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.alt_label = alternative_label
        self.is_opaque = is_opaque

    def make_opaque(self) -> None:
        self.is_opaque = True

class PointLoad():
    """PointLoad class represents a point load in 2D space defined by its location and properties."""
    def __init__(self, point: Point, value: float, unit: str = 'kN', dxdy:tuple[float,float]=(0,1), anglelabel=False, anglelabelflip:bool=False, labelpos:tuple[str,str]=('top', 'center'), alternative_label: str = None, color: str = 'red', is_opaque: bool = False) -> None:
        self.value = value
        self.dx = dxdy[0]
        self.dy = dxdy[1]
        self.unit = unit
        self.x = point.x
        self.y = point.y
        self.anglelabel = anglelabel
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.labelflip = anglelabelflip
        self.alt_label = alternative_label
        self.color = resolve_force_color(color)
        self.is_opaque = is_opaque

    def make_opaque(self) -> None:
        self.is_opaque = True
        
class DistributedLoad():
    """DistributedLoad class represents a distributed load in 2D space defined by its location and properties."""
    def __init__(self, begin_point: Point, end_point: Point, begin_value: float, end_value: float = None, unit: str = 'kN/m', angle: float=90, n_arrow = 6, labelpos:tuple[str,str]=('top', 'center'), labelpos_end:tuple[str,str]=None, alternative_label_begin: str = None, alternative_label_end: str = None, color: str = 'red', is_opaque: bool = False) -> None:
        self.begin_value = begin_value
        self.end_value = end_value if end_value is not None else begin_value
        self.unit = unit
        self.begin = begin_point
        self.end = end_point
        self.angle = angle
        self.n_arrow = n_arrow
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        if labelpos_end is not None:
            self.labelx_end = labelpos_end[1]
            self.labely_end = labelpos_end[0]
        else:
            self.labelx_end = labelpos[1]
            self.labely_end = labelpos[0]
        self.alt_label_begin = alternative_label_begin
        self.alt_label_end = alternative_label_end
        self.color = resolve_force_color(color)
        self.is_opaque = is_opaque

    def make_opaque(self) -> None:
        self.is_opaque = True

class Moment():
    """Moment class represents a moment in 2D space defined by its location and properties."""
    def __init__(self, point: Point, value: float=None, unit: str = 'kNm', clock_wise: bool = True, angle: float = 0.0, labelpos:tuple[str,str]=('top', 'center'), alternative_label: str = None, color: str = 'red', is_opaque: bool = False) -> None:
        if value is not None and value < 0:
            self.value = -value
            self.clock_wise = not clock_wise
        else:
            self.value = value
            self.clock_wise = clock_wise 
        self.point = point
        self.unit = unit
        self.angle = angle
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.alt_label = alternative_label
        self.color = resolve_force_color(color)
        self.is_opaque = is_opaque

    def make_opaque(self) -> None:
        self.is_opaque = True

class TwistingMoment(PointLoad):
    """TwistingMoment class represents a twisting moment in 2D space defined by its location and properties."""
    def __init__(self, point: Point, value: float, unit: str = 'kNm', dxdy:tuple[float,float]=(0,1), anglelabel=False, anglelabelflip:bool=False, labelpos:tuple[str,str]=('top', 'center'), alternative_label: str = None, color: str = 'red', is_opaque: bool = False) -> None:
        self.value = value
        self.dx = dxdy[0]
        self.dy = dxdy[1]
        self.unit = unit
        self.x = point.x
        self.y = point.y
        self.anglelabel = anglelabel
        self.labelx = labelpos[1]
        self.labely = labelpos[0]
        self.labelflip = anglelabelflip
        self.alt_label = alternative_label
        self.color = resolve_force_color(color)
        self.is_opaque = is_opaque

class Length():
    """Length class represents a length in 2D space defined by its location and properties."""
    def __init__(self, point1: Point, point2: Point, ax: str='x', xpos: str = 'bottom', ypos: str ='left', alternative_label:str = None, is_opaque: bool = False) -> None:
        self.point1 = point1
        self.point2 = point2
        self.pos = ax
        self.xpos = xpos
        self.ypos = ypos
        self.altlabel = alternative_label
        self.is_opaque = is_opaque

    def make_opaque(self) -> None:
        self.is_opaque = True

def resolve_force_color(color: str) -> str:
    """Turns a color name into a hexcode for plotting.

    Args:
        color (str): Color name to resolve to hexcode. Must be one of 'red', 'green', 'blue'.

    Raises:
        ValueError: If color is not in FORCE_COLORS

    Returns:
        str: hexcode for given color, if not in FORCE_COLORS raises ValueError
    """
    if color not in FORCE_COLORS:
        raise ValueError("Force color must be one of: 'red', 'green', 'blue'")
    return FORCE_COLORS[color]

def rotate_point(beam: Beam, x: float, y: float) -> tuple[float, float]:
    """Rotates a point (x, y) around the start point of a beam by the angle of the beam.

    Args:
        beam (Beam): The beam around which the point will be rotated.
        x (float): The x-coordinate of the point to be rotated, relative to the start point of the beam.
        y (float): The y-coordinate of the point to be rotated, relative to the start point of the beam.

    Returns:
        tuple[float, float]: The coordinates of the rotated point.
    """
    a = np.radians(beam.angle)
    return (beam.x1 + x*np.cos(a) - y*np.sin(a),
            beam.y1 + x*np.sin(a) + y*np.cos(a))

def rotate_point_reverse(beam: Beam, x: float, y: float) -> tuple[float, float]:
    """Rotates a point (x, y) around the start point of a beam by the angle of the beam in the opposite direction.

    Args:
        beam (Beam): The beam around which the point will be rotated.
        x (float): The x-coordinate of the point to be rotated, relative to the start point of the beam.
        y (float): The y-coordinate of the point to be rotated, relative to the start point of the beam.

    Returns:
        tuple[float, float]: The coordinates of the rotated point.
    """
    a = np.radians(beam.angle)
    return (beam.x1 - x*np.cos(a) + y*np.sin(a),
            beam.y1 - x*np.sin(a) - y*np.cos(a))

def parabole(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> list[tuple[float, float]]:
    """Calculates the points on a parabola defined by three points.

    Args:
        x1 (float): The x-coordinate of the first point.
        y1 (float): The y-coordinate of the first point.
        x2 (float): The x-coordinate of the second point.
        y2 (float): The y-coordinate of the second point.
        x3 (float): The x-coordinate of the third point.
        y3 (float): The y-coordinate of the third point.

    Returns:
        list[tuple[float, float]]: A list of points on the parabola.
    """
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
    return parlist


class Structure():
    """Structure class represents a collection of points, beams, supports, loads, and other elements in 2D space."""
    def __init__(self) -> None:
        self._points = set()
        self._beams = set()
        self._pointloads = set()
        self._moments = set()
        self._twistingmoments = set()
        self._distributedloads = set()
        self._hinges = set()
        self._fixedsupports = set()
        self._pinnedsupports = set()
        self._rollersupports = set()
        self._rotationsprings = set()
        self._translationsprings = set()
        self._lengths = set()

    def add_point(self, *points: Point) -> None:
        for point in points:
            self._points.add(point)

    def add_hinge(self, *points: Point) -> None:
        for point in points:
            self._hinges.add(point)
            self._points.add(point)

    def add_beam(self, *beams: Beam) -> None:
        for beam in beams:
            self._beams.add(beam)
            self.add_point(beam.begin)
            self.add_point(beam.end)

    def add_deformedbeam(self, *deformedbeams: DeformedBeam) -> None:
        for deformedbeam in deformedbeams:
            self._deformedbeams.add(deformedbeam)
            self.add_point(deformedbeam.begin)
            self.add_point(deformedbeam.end)

    def add_support(self, *supports: Support) -> None:
        for support in supports:
            self.add_point(support.loc)
            if support._type == 'fixed':
                self._fixedsupports.add(support)
            if support._type == 'roller':
                self._rollersupports.add(support)
            if support._type == 'pinned':
                self._pinnedsupports.add(support)

    def add_pointload(self, *pointloads: PointLoad) -> None:
        for pointload in pointloads:
            self._pointloads.add(pointload)

    def add_moment(self, *moments: Moment) -> None:
        for moment in moments:
            self._moments.add(moment)

    def add_twistingmoment(self, *twistingmoments: TwistingMoment) -> None:
        for twistingmoment in twistingmoments:
            self._twistingmoments.add(twistingmoment)

    def add_distributedload(self, *distributedloads: DistributedLoad) -> None:
        for distributedload in distributedloads:
            self._distributedloads.add(distributedload)

    def add_rotationspring(self, *rotationsprings: RotationSpring) -> None:
        for rotationspring in rotationsprings:
            self._rotationsprings.add(rotationspring)

    def add_translationspring(self, *translationsprings: TranslationSpring) -> None:
        for translationspring in translationsprings:
            self._translationsprings.add(translationspring)

    def add_length(self, *lengths: Length) -> None:
        for length in lengths:
            self._lengths.add(length)

    def xminmax(self) -> tuple[float, float]:
        """Returns the minimum and maximum x-coordinates of all points in the structure."""
        xmin = 999 
        xmax = -999
        for p in self._points:
            xmin = min(p.x, xmin)
            xmax = max(p.x, xmax)
        return xmin, xmax

    def yminmax(self) -> tuple[float, float]:
        """Returns the minimum and maximum y-coordinates of all points in the structure."""
        ymin = 999
        ymax = -999
        for p in self._points:
            ymin = min(p.y, ymin)
            ymax = max(p.y, ymax)
        return ymin, ymax
    
    def opaque_list(self, *args) -> None:
        for arg in args:
            arg.make_opaque()

    def dashed_list(self, *args) -> None:
        for arg in args:
            arg.make_dashed()

def plot(structure: Structure, name: str = None, format: str = 'svg', is_seed: bool = True) -> None:
    """Plots the given structure using matplotlib.

    Args:
        structure (Structure): The structure to plot.
        name (str, optional): The name of the plot. Defaults to None.
        format (str, optional): The format of the plot. Defaults to 'svg'.
        is_seed (bool, optional): Whether to use a seed for the plot. Defaults to True.

    Returns:
        None
    """
    plt.plot([0,0],[0,0],color='black',linewidth=2)
    axs = plt.gca()
    axs.axis('equal')
    axs.axis('off')
    xmin, xmax = structure.xminmax()
    xlength = max(1, xmax - xmin)
    ymin, ymax = structure.yminmax()
    ylength = max(1, ymax - ymin)
    #print(xmin, xmax, ymin, ymax)
    # scaler = (np.sqrt(xlength*ylength) - np.sqrt(3*1)) / (np.sqrt(20*6) - np.sqrt(3*1))
    scaler = (max(xlength,ylength) - max(3,1)) / (max(20,6) - max(3,1))
    #print(scaler)
    # scaler = 2*(xlength*ylength - 3*1) / (20*6 - 3*1)
    # plt.xlim(xmin - 2, xmax + 2)
    # plt.ylim(ymin - 2, ymax + 2)
    #axs.margins(0.2)

    def drawdeformedbeam(deformedbeam: DeformedBeam) -> None:
        """draws a deformed beam on the plot using matplotlib.

        Args:
            deformedbeam (DeformedBeam): The deformed beam to draw.
        """
        alpha = 1.0 if not deformedbeam.is_opaque else 0.5
        linestyle = (5, (8, 3)) if deformedbeam.is_dashed else 'solid'
        linewidth = 1 if deformedbeam.is_dashed else 2
        x1, y1 = deformedbeam.begin.x, deformedbeam.begin.y
        x2, y2 = deformedbeam.end.x, deformedbeam.end.y
        a = np.linspace(0, deformedbeam.length, 1000) # Replace with your desired range and number of points
        f = deformedbeam.function
        y_beam = np.full_like(a, f(a))
        beam = Beam(deformedbeam.begin, deformedbeam.end)
        xy_list = [rotate_point(beam, xi, yi) for xi, yi in zip(a, y_beam)]
        plt.plot([x for x, y in xy_list], [y for x, y in xy_list], color='black', linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    def drawbeam(beam: Beam) -> None:
        """draws a beam on the plot using matplotlib.

        Args:
            beam (Beam): The beam to draw.
        """
        if isinstance(beam, DeformedBeam):
            drawdeformedbeam(beam)
        else:
            alpha = 1.0 if not beam.is_opaque else 0.5
            linestyle = (5, (8, 3)) if beam.is_dashed else 'solid'
            linewidth = 1 if beam.is_dashed else 2
            if isinstance(beam, ParabolicBeam):
                # calculate the rotated coordinates of the start, middle and end points of the parabolic beam
                x1_r, y1_r = 0, 0
                x2_r, y2_r = beam.length, 0
                xm_r, ym_r = 0.5*beam.length , beam.ym
                xylist = parabole(x1_r, y1_r, xm_r, ym_r, x2_r, y2_r)
                xy_list_rotated = [rotate_point(beam, x, y) for x, y in xylist]
                plt.plot([x for x, y in xy_list_rotated], [y for x, y in xy_list_rotated], color='black', linewidth=linewidth, alpha=alpha, linestyle=linestyle)
            else:
                plt.plot([beam.x1, beam.x2], [beam.y1, beam.y2], color='black', linewidth=linewidth, alpha=alpha, linestyle=linestyle)
            amin = 0.05
            amax = 0.4
            a = amin + scaler * (amax - amin)
            rmin = 0.15*0.3
            rmax = 0.6*0.3
            r = rmin + scaler * (rmax - rmin)
            for hinge, is_begin in beam.hinges:
                h = Circle([hinge.x - (1 - 2*int(is_begin))* a*np.cos(np.radians(beam.angle)), hinge.y - (1 - 2*int(is_begin))* a*np.sin(np.radians(beam.angle))], radius=r, facecolor='white',edgecolor='black', zorder=linewidth, alpha=alpha, linestyle=linestyle)
                axs.add_patch(h)

            midx = (beam.x1+beam.x2)/2
            midy = (beam.y1+beam.y2)/2
            if isinstance(beam, ParabolicBeam):
                midx, midy = xy_list_rotated[1]

            lmin = 0.5
            lmax = 0.5
            l = lmin + scaler * (lmax - lmin)
            if beam.label is not None:
                x = midx
                if beam.labelx == 'left':
                    x -= l
                if beam.labelx == 'right':
                    x += l

                y = midy
                if beam.labely == 'top':
                    y += l
                if beam.labely == 'bottom':
                    y -= l
                axs.annotate(text=beam.label, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)
                
            if beam.anglelabel:
                dx1 = (1 - 2*int(beam.labelflip)) * (beam.x2 - beam.x1) / 4
                dy1 = (1 - 2*int(beam.labelflip)) * (beam.y2 - beam.y1) / 4
                dx = int(abs(beam.x2 - beam.x1)*100) 
                dy = int(abs(beam.y2 - beam.y1)*100) 
                gcd = np.gcd(dx, dy)
                dx = int(dx/gcd)
                dy = int(dy/gcd)
                plt.plot([midx-dx1/2,midx+dx1/2,midx+dx1/2],[midy-dy1/2,midy-dy1/2,midy+dy1/2],linewidth=1,color="black", alpha=alpha)
                umin = 0.2
                umax = 0.6
                u = umin + scaler*(umax-umin)
                axs.annotate(text=str(dy), xy=(midx + 0.5*dx1 + u*dx1/abs(dx1), midy), ha='center',va='center', fontname='Times New Roman', alpha=alpha)
                axs.annotate(text=str(dx), xy=(midx, midy - 0.5*dy1 - u*dy1/abs(dy1)), ha='center',va='center', fontname='Times New Roman', alpha=alpha)

    for b in structure._beams:
        drawbeam(b)

    def drawpoint(point: Point) -> None:
        """draws a point on the plot using matplotlib

        Args:
            point (Point): the point to draw
        """
        alpha = 1.0 if not point.is_opaque else 0.5
        lmin = 0.2
        lmax = 0.5
        l = lmin + scaler * (lmax - lmin)
        x = point.x
        if point.labelx == 'left':
            x -= l
        if point.labelx == 'right':
            x += l

        y = point.y
        if point.labely == 'top':
            y += l
        if point.labely == 'bottom':
            y -= l 

        axs.annotate(text=point.label, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)

    for p in structure._points:
        drawpoint(p)

    def drawhinge(hinge: Point) -> None:
        """draws a hinge on the plot using matplotlib

        Args:
            hinge (Point): The hinge to draw, represented as a Point object with x and y coordinates.
        """
        alpha = 0.5 if hinge.is_opaque else 1.0
        linestyle = 'solid' if not hinge.is_dashed else (0, (4, 2))
        linewidth = 2 if hinge.is_dashed else 1
        rmin = 0.15
        rmax = 0.6
        r = rmin + scaler*(rmax-rmin)
        h = Circle([hinge.x,hinge.y], radius=r*0.3, facecolor='white',edgecolor='black', zorder=linewidth, alpha=alpha, linestyle=linestyle)
        axs.add_patch(h)

    for h in structure._hinges:
        drawhinge(h)

    def drawmoment(moment: Moment) -> None:
        """draws a moment on the plot using matplotlib

        Args:
            moment (Moment): The moment to draw, represented as a Moment object with properties such as point, value, unit, angle, and color.
        """
        alpha = 1.0 if not moment.is_opaque else 0.5
        angle = moment.angle
        rmin = 0.5
        rmax = 2.5
        radius = rmin + scaler * (rmax - rmin)
        m = Arc((moment.point.x, moment.point.y), width=radius, height = radius, angle=angle, theta1=0, theta2=150, color=moment.color, alpha=alpha, linewidth=2)
        axs.add_patch(m)
            
        arrowhead = FancyArrow(
                        moment.point.x + radius * 0.5 * np.cos(np.radians(1*(angle + 10+(1-int(moment.clock_wise))*150))),
                        moment.point.y + radius * 0.5 * np.sin(np.radians(1*(angle + 10+(1-int(moment.clock_wise))*150))),
                        (2*int(moment.clock_wise)-1)* 0.1 * np.cos(np.radians(angle + 10 + (1-int(moment.clock_wise))*150 - 90)), #moet groter worden met de radius
                        (2*int(moment.clock_wise)-1)* 0.1 * np.sin(np.radians(angle + 10 + (1-int(moment.clock_wise))*150 - 90)),
                        width=0.02*radius/rmin,
                        head_width=0.1*radius/rmin,
                        head_length=0.15*radius/rmin,
                        color=moment.color,
                        alpha=alpha,
                        length_includes_head=True
                    )
        axs.add_patch(arrowhead)
        umin = 0.2
        umax = 0.6
        u = umin + scaler * (umax-umin)
        x = moment.point.x 
        if moment.labelx == 'left':
            x -= 0.5*radius + u 
        if moment.labelx == 'right':
            x += 0.5*radius + u

        y = moment.point.y
        if moment.labely == 'top':
            y += 0.5*radius + u
        if moment.labely == 'bottom':
            y -= 0.5*radius + u 
            
        if moment.alt_label is not None:
            axs.annotate(text=moment.alt_label, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)
        elif moment.value is not None:
            axs.annotate(text=str(moment.value) + ' ' + moment.unit, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)

    for h in structure._moments:
        drawmoment(h)

    def drawtwistingmoment(twistingmoment: TwistingMoment) -> None:
        """draws a twisting moment on the plot using matplotlib

        Args:
            twistingmoment (TwistingMoment): The twisting moment to draw, represented as a TwistingMoment object with properties such as point, value, unit, angle, and color.
        """
        alpha = 1.0 if not twistingmoment.is_opaque else 0.5
        lmin = 0.8
        lmax = 2.5
        length = lmin + scaler*(lmax-lmin) # afhankelijk van andere pointloads en totale grootte van de structure
        if twistingmoment.anglelabel:
            length = 2 * length # if label is wanted, extend the arrow so there is room for the label
        tip = (twistingmoment.x, twistingmoment.y)
        ddx = twistingmoment.dx * length / np.sqrt(twistingmoment.dx**2 + twistingmoment.dy**2)
        ddy = twistingmoment.dy * length / np.sqrt(twistingmoment.dx**2 + twistingmoment.dy**2)
        head_dist = 0.15*length

        norm = np.sqrt(ddx**2 + ddy**2)
        ux = ddx / norm
        uy = ddy / norm

        second_tip = (tip[0] + head_dist * ux, tip[1] + head_dist * uy)
        start = (twistingmoment.x + ddx, twistingmoment.y + ddy) # depends on angle and wanted length

        main_arrow = FancyArrow(
                                start[0],
                                start[1],
                                dx=second_tip[0] - start[0],
                                dy=second_tip[1] - start[1],
                                width=0.04*length,
                                head_width=0.15*length,
                                head_length=0.15*length,
                                length_includes_head=True,
                                color=twistingmoment.color,
                                alpha=alpha,
                            )

        axs.add_patch(main_arrow)

        # Direction from second_tip to tip
        dx = tip[0] - second_tip[0]
        dy = tip[1] - second_tip[1]
        norm = np.sqrt(dx**2 + dy**2)

        ux = dx / norm
        uy = dy / norm

        # Perpendicular direction
        px = -uy
        py = ux

        # Triangle dimensions
        head_length = 0.15 * length
        head_width = 0.15 * length

        # Tip point
        tip_point = np.array(tip)

        # Back of triangle
        base_center = tip_point - head_length * np.array([ux, uy])

        # Two base corners
        left = base_center + (head_width / 2) * np.array([px, py])
        right = base_center - (head_width / 2) * np.array([px, py])

        triangle = Polygon(
            [tip_point, left, right],
            closed=True,
            facecolor=twistingmoment.color,
            edgecolor=twistingmoment.color,
            alpha=alpha
        )

        axs.add_patch(triangle)        

        # extra_head = FancyArrow(second_tip[0],  
        #                         second_tip[1], 
        #                         dx=tip[0]-second_tip[0], 
        #                         dy=tip[1]-second_tip[1],
        #                             width=1e-6,
        #                             head_width=0.15*length,
        #                             head_length=0.15*length,
        #                             length_includes_head=True,
        #                             facecolor=twistingmoment.color,
        #                             edgecolor=twistingmoment.color,
        #                             alpha=alpha)

        # axs.add_patch(extra_head)

        umin = 0.1
        umax = 0.1
        u = umin + scaler * (umax - umin)
        x = start[0]
        if twistingmoment.labelx == 'left':
            x -= 2*u * length
        if twistingmoment.labelx == 'right':
            x += 2*u * length

        y = start[1]
        if twistingmoment.labely == 'top':
            y += u * length
        if twistingmoment.labely == 'bottom':
            y -= u * length 
        if twistingmoment.value is not None:
            axs.annotate(text=str(twistingmoment.value) + ' ' + twistingmoment.unit, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)
        else:
            axs.annotate(text=twistingmoment.alt_label, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)

        if twistingmoment.anglelabel: 
            dx1 = (1 - 2*int(twistingmoment.labelflip)) * ddx/3
            dy1 = (1 - 2*int(twistingmoment.labelflip)) * ddy/3
            midx = (tip[0] + start[0])/2
            midy = (tip[1] + start[1])/2
            dx = int(abs(twistingmoment.dx)*100)
            dy = int(abs(twistingmoment.dy)*100)
            gcd = np.gcd(dx, dy)
            dx = int(dx/gcd)
            dy = int(dy/gcd)
            umin = 0.2
            umax = 0.6
            u = umin + scaler*(umax-umin)
            plt.plot([midx-dx1/2,midx+dx1/2,midx+dx1/2],[midy-dy1/2,midy-dy1/2,midy+dy1/2],linewidth=1,color="black")
            axs.annotate(text=str(dy), xy=(midx + 0.5*dx1 + u*dx1/abs(dx1), midy), ha='center',va='center', fontname='Times New Roman', alpha=alpha)
            axs.annotate(text=str(dx), xy=(midx, midy - 0.5*dy1 - u*dy1/abs(dy1)), ha='center',va='center', fontname='Times New Roman', alpha=alpha)

    for p in structure._twistingmoments:
        drawtwistingmoment(p)        

    def drawpointload(pointload: PointLoad) -> None:
        """draws a pointload on the plot using matplotlib

        Args:
            pointload (PointLoad): The pointload to draw, represented as a PointLoad object with properties such as x, y, dx, dy, value, unit, and color.
        """
        alpha = 1.0 if not pointload.is_opaque else 0.5
        lmin = 0.8
        lmax = 2.5
        length = lmin + scaler*(lmax-lmin) # afhankelijk van andere pointloads en totale grootte van de structure
        if pointload.anglelabel:
            length = 2 * length # if label is wanted, extend the arrow so there is room for the label
        tip = (pointload.x, pointload.y)
        ddx = pointload.dx * length / np.sqrt(pointload.dx**2 + pointload.dy**2)
        ddy = pointload.dy * length / np.sqrt(pointload.dx**2 + pointload.dy**2)
        start = (pointload.x + ddx, pointload.y + ddy) # depends on angle and wanted length

        axs.annotate(text='', xy=tip, xytext=start, arrowprops=dict(arrowstyle='simple',color=pointload.color, alpha=alpha))

        umin = 0.1
        umax = 0.1
        u = umin + scaler * (umax - umin)
        x = start[0]
        if pointload.labelx == 'left':
            x -= 2*u * length
        if pointload.labelx == 'right':
            x += 2*u * length

        y = start[1]
        if pointload.labely == 'top':
            y += u * length
        if pointload.labely == 'bottom':
            y -= u * length 
        if pointload.value is not None:
            axs.annotate(text=str(pointload.value) + ' ' + pointload.unit, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)
        else:
            axs.annotate(text=pointload.alt_label, xy=(x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)

        if pointload.anglelabel: 
            dx1 = (1 - 2*int(pointload.labelflip)) * ddx/3
            dy1 = (1 - 2*int(pointload.labelflip)) * ddy/3
            midx = (tip[0] + start[0])/2
            midy = (tip[1] + start[1])/2
            dx = int(abs(pointload.dx)*100)
            dy = int(abs(pointload.dy)*100)
            gcd = np.gcd(dx, dy)
            dx = int(dx/gcd)
            dy = int(dy/gcd)
            umin = 0.2
            umax = 0.6
            u = umin + scaler*(umax-umin)
            plt.plot([midx-dx1/2,midx+dx1/2,midx+dx1/2],[midy-dy1/2,midy-dy1/2,midy+dy1/2],linewidth=1,color="black")
            axs.annotate(text=str(dy), xy=(midx + 0.5*dx1 + u*dx1/abs(dx1), midy), ha='center',va='center', fontname='Times New Roman', alpha=alpha)
            axs.annotate(text=str(dx), xy=(midx, midy - 0.5*dy1 - u*dy1/abs(dy1)), ha='center',va='center', fontname='Times New Roman', alpha=alpha)

    for p in structure._pointloads:
        drawpointload(p)

    def drawdistributedload(dload: DistributedLoad) -> None:
        """draws a distributed load on the plot using matplotlib

        Args:
            dload (DistributedLoad): The distributed load to draw, represented as a DistributedLoad object with properties such as begin, end, begin_value, end_value, angle, n_arrow, and color.
        """
        alpha = 1.0 if not dload.is_opaque else 0.5
        lmin = 0.6
        lmax = 2.5
        length_mid = lmin + scaler * (lmax - lmin)
        beam_length = dload.begin.distance_to(dload.end)
        beam_angle = dload.end.angle(dload.begin)
        n_arrow = dload.n_arrow
        dist = beam_length/(n_arrow - 1)
        v_mid = (dload.begin_value + dload.end_value)/2
        length = length_mid / v_mid * np.linspace(dload.begin_value, dload.end_value, n_arrow)
        # plot line
        plt.plot([dload.begin.x + 0.95 * length[0] * np.cos(np.radians(dload.angle)), dload.end.x + 0.95 * length[-1] * np.cos(np.radians(dload.angle))], 
                 [dload.begin.y + 0.95 * length[0] * np.sin(np.radians(dload.angle)), dload.end.y + 0.95 * length[-1] * np.sin(np.radians(dload.angle))], 
                 color=dload.color, linewidth=2, alpha=alpha)
        # plot arrows
        for i in range(n_arrow):
            tip = (dload.begin.x + i * dist * np.cos(np.radians(beam_angle)), 
                   dload.begin.y + i * dist * np.sin(np.radians(beam_angle)))
            start = (dload.begin.x + i * dist * np.cos(np.radians(beam_angle)) + length[i] * np.cos(np.radians(dload.angle)), 
                     dload.begin.y + i * dist * np.sin(np.radians(beam_angle)) + length[i] * np.sin(np.radians(dload.angle)))
            if np.sqrt((tip[0]-start[0])**2+(tip[1]-start[1])**2) > 0.1:
                axs.annotate(text='', xy=tip, xytext=start, arrowprops=dict(arrowstyle='simple',color=dload.color, alpha=alpha))

        # display text at begin point
        x = dload.begin.x + length[0] * np.cos(np.radians(dload.angle))       
        if dload.labelx == 'left':
            x -= 0.4 * length_mid
        if dload.labelx == 'right':
            x += 0.4 * length_mid 

        y = dload.begin.y +  length[0] * np.sin(np.radians(dload.angle)) 
        if dload.labely == 'top':
            y += 0.2 * length_mid 
        if dload.labely == 'bottom':
            y -= 0.2 * length_mid 
        
        if dload.alt_label_begin is None:
            axs.annotate(text=str(dload.begin_value) + ' ' + dload.unit, xy = (x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)
        else:
            axs.annotate(text=dload.alt_label_begin, xy = (x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)

        # display text at end point
        x = dload.end.x + length[-1] * np.cos(np.radians(dload.angle))       
        if dload.labelx_end == 'left':
            x -= 0.4 * length_mid
        if dload.labelx_end == 'right':
            x += 0.4 * length_mid 

        y = dload.end.y +  length[-1] * np.sin(np.radians(dload.angle)) 
        if dload.labely_end == 'top':
            y += 0.2 * length_mid 
        if dload.labely_end == 'bottom':
            y -= 0.2 * length_mid 
        
        if dload.alt_label_end is None:
            axs.annotate(text=str(dload.end_value) + ' ' + dload.unit, xy = (x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)
        else:
            axs.annotate(text=dload.alt_label_end, xy = (x,y), ha='center', va='center', fontname='Times New Roman', alpha=alpha)

    for d in structure._distributedloads:
        drawdistributedload(d)

    def drawfixedsupport(support: Support) -> None:
        """draws a fixed support on the plot using matplotlib

        Args:
            support (Support): The fixed support to draw, represented as a Support object with properties such as x, y, angle, and is_opaque.
        """
        alpha = 1.0 if not support.is_opaque else 0.5
        angle = support.angle
        #angle = 30
        amin = 0.2
        amax = 0.8
        a = amin + scaler * (amax-amin)
        thickness = 15
        size = 100
        size2 = 200
        pattern = np.zeros((size, size2))
        for j in range(size):
            for i in range(size2):
                if ((i + j) // thickness) % 2 == 0:
                    pattern[j, i] = 1  
        pattern_rotated = rotate(pattern, -angle, reshape=True, order=0, mode='constant', cval=0)
        pattern = np.flipud(pattern)
        # Plot the pattern using imshow()
        original_width = 2*a
        original_height = a
        new_width = abs(original_width * np.cos(np.radians(angle))) + abs(original_height * np.sin(np.radians(angle)))
        new_height = abs(original_width * np.sin(np.radians(angle))) + abs(original_height * np.cos(np.radians(angle)))

        axs.imshow(pattern_rotated, 
                   cmap="gray_r", 
                   extent=[support.x - 0.5*(new_width - a*np.sin(np.radians(angle))), 
                    support.x + 0.5*(new_width + a*np.sin(np.radians(angle))), 
                    support.y - 0.5*(new_height + a*np.cos(np.radians(angle))), 
                    support.y + 0.5*(new_height - a*np.cos(np.radians(angle)))], 
                   aspect='equal', 
                   origin='lower',
                   alpha=alpha)
        #axs.set_xlim(xmin - 2, xmax + 2)
        #axs.set_ylim(ymin - 2, ymax + 2)
        #axs.margins(0.2)
        # Draw a line next to the rectangle
        axs.plot([support.x - a*np.cos(np.radians(angle)), 
                support.x + a*np.cos(np.radians(angle))], 
                [support.y - a*np.sin(np.radians(angle)), 
                support.y + a*np.sin(np.radians(angle))], 
                color='black', linewidth=2, alpha=alpha)
        
    for f in structure._fixedsupports:
        drawfixedsupport(f)

    def drawrollersupport(support: Support) -> None:
        """draws a roller support on the plot using matplotlib

        Args:
            support (Support): The roller support to draw, represented as a Support object with properties such as x, y, angle, and is_opaque.
        """
        alpha = 1.0 if not support.is_opaque else 0.5
        basetriangle = np.array([[0,0], [-0.67, -1], [0.67, -1], [0,0]])
        baseline1 = np.array([[-1, -1], [1, -1]])
        baseline2 = np.array([[-1, -1.2], [1, -1.2]])
        # scaling
        smin = 0.2
        smax = 0.8
        scaling = smin + scaler * (smax - smin)
        scaledtriangle = basetriangle @ np.array([[scaling, 0], [0, scaling]]).T
        scaledline1 = (baseline1 - (0,-1)) @ np.array([[scaling, 0], [0, scaling]]).T + (0,-1*scaling)
        scaledline2 = (baseline2 - (0,-1.2)) @ np.array([[scaling, 0], [0, scaling]]).T + (0,-1.2*scaling)
        # rotating
        angle = np.radians(support.angle) # in radialen
        rotatedtriangle = scaledtriangle @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotatedline1 = scaledline1 @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotatedline2 = scaledline2 @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # put support at the right point
        shiftedtriangle = rotatedtriangle + (support.x,support.y) # move to wanted point
        shiftedline1 = rotatedline1 + (support.x,support.y)
        shiftedline2 = rotatedline2 + (support.x,support.y)
        # make patches and add them
        triangle = Polygon(shiftedtriangle, facecolor = '#00CC99',edgecolor='black', alpha=alpha)
        line1 = Polygon(shiftedline1, fill=False, edgecolor='black', linewidth=1, alpha=alpha)
        line2 = Polygon(shiftedline2, fill=False, edgecolor='black', linewidth=1, alpha=alpha)
        axs.add_patch(triangle)
        axs.add_patch(line1)
        axs.add_patch(line2)
    
    for r in structure._rollersupports:
        drawrollersupport(r) 

    def drawpinnedsupport(support: Support) -> None:
        """draws a pinned support on the plot using matplotlib

        Args:
            support (Support): The pinned support to draw, represented as a Support object with properties such as x, y, angle, and is_opaque.
        """
        alpha = 1.0 if not support.is_opaque else 0.5
        basetriangle = np.array([[0,0], [-0.67, -1], [0.67, -1], [0,0]])
        baseline1 = np.array([[-1, -1], [1, -1]])
        # scaling
        smin = 0.2
        smax = 0.8
        scaling = smin + scaler * (smax - smin)
        scaledtriangle = basetriangle @ np.array([[scaling, 0], [0, scaling]]).T
        scaledline1 = (baseline1 - (0,-1)) @ np.array([[scaling, 0], [0, scaling]]).T + (0,-1*scaling)
        # rotating
        angle = np.radians(support.angle) # in radialen
        rotatedtriangle = scaledtriangle @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotatedline1 = scaledline1 @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # put support at the right point
        shiftedtriangle = rotatedtriangle + (support.x,support.y) # move to wanted point
        shiftedline1 = rotatedline1 + (support.x,support.y)
        # make patches and add them
        triangle = Polygon(shiftedtriangle, facecolor = '#00CC99',edgecolor='black', alpha=alpha)
        line1 = Polygon(shiftedline1, fill=False, edgecolor='black', linewidth=1, alpha=alpha)
        axs.add_patch(triangle)
        axs.add_patch(line1)

    for p in structure._pinnedsupports:
        drawpinnedsupport(p)

    def drawrotationspring(rspring: RotationSpring) -> None:
        """draws a rotation spring on the plot using matplotlib

        Args:
            rspring (RotationSpring): The rotation spring to draw, represented as an object with properties such as x, y, and is_opaque.
        """
        alpha = 1.0 if not rspring.is_opaque else 0.5
        theta = np.radians(np.linspace(2,360*2,1000))
        rmin = 0.025
        rmax = 0.08
        r = theta**0.7 * (rmin + scaler * (rmax-rmin) )
        x_2 = r*np.cos(theta) + rspring.x
        y_2 = r*np.sin(theta) + rspring.y
        plt.plot(x_2,y_2, color='black', linewidth=1, alpha=alpha)

    for r in structure._rotationsprings:
        drawrotationspring(r)

    def drawtranslationspring(tspring: TranslationSpring) -> None:
        """draws a translation spring on the plot using matplotlib

        Args:
            tspring (TranslationSpring): The translation spring to draw, represented as an object with properties such as x1, x2, y1, y2, and is_opaque.
        """
        alpha = 1.0 if not tspring.is_opaque else 0.5
        x1, x2 = tspring.x1, tspring.x2
        y1, y2 = tspring.y1, tspring.y2
        n_spikes = 10
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        spike_length = length / n_spikes
        spike_x = np.linspace(0, length, n_spikes*2+1)
        spike_y = np.linspace(0, 0, n_spikes*2+1)
        for i in range(len(spike_x)):
            if (i+3) % 4 == 0:
                spike_y[i] += 2*spike_length
            if (i+1) % 4 == 0:
                spike_y[i] -= 2*spike_length
        spike_x_rotated, spike_y_rotated = zip(*(rotate_point(Beam(tspring.startpoint, tspring.endpoint), xi, yi)
                                            for xi, yi in zip(spike_x, spike_y)))
        plt.plot(spike_x_rotated, spike_y_rotated, color='black', linewidth=1, alpha=alpha)

    for t in structure._translationsprings:
        drawtranslationspring(t)
    
    def drawlength(length: Length) -> None:
        """draws a length on the plot using matplotlib

        Args:
            length (Length): The length to draw, represented as a Length object with properties such as point1, point2, pos, xpos, ypos, altlabel, and is_opaque.
        """
        alpha = 1.0 if not length.is_opaque else 0.5
        if length.pos == 'x':
            x1 = length.point1.x
            x2 = length.point2.x
            uymin = 0.6
            uymax = 2.2
            uy = uymin + scaler * (uymax - uymin)
            if length.xpos == 'bottom':
                y = ymin - uy
            else:
                y = ymax + uy
            text = str(round(abs(x2 - x1), 1)) + ' m' if length.altlabel is None else length.altlabel
            axs.annotate(text='', xy=(x1,y), xytext=(x2,y), arrowprops=dict(arrowstyle='<->',shrinkA=0,shrinkB=0, alpha=alpha))
            umin = 0.2
            umax = 1
            u = umin +scaler*(umax-umin)
            axs.annotate(text=text,xy=((x2+x1)/2,y+u),ha='center',va='top', fontname='Times New Roman', alpha=alpha)
        else:
            y1 = length.point1.y
            y2 = length.point2.y
            uxmin = 0.8
            uxmax = 2
            ux = uxmin + scaler * (uxmax - uxmin)
            if length.ypos == 'left':
                x = xmin - ux
            else:
                x = xmax + ux
            text = str(round(abs(y2 - y1), 1)) + ' m' if length.altlabel is None else length.altlabel
            axs.annotate(text='', xy=(x,y1), xytext=(x,y2), arrowprops=dict(arrowstyle='<->',shrinkA=0,shrinkB=0, alpha=alpha))
            umin = 0.3
            umax = 1
            u = umin +scaler*(umax-umin)
            axs.annotate(text=text,xy=(x+u, (y2+y1)/2),ha='center',va='center', fontname='Times New Roman', alpha=alpha)

    for l in structure._lengths:
        drawlength(l)

    axs.use_sticky_edges = False
    axs.autoscale()
    fig = plt.gcf()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    def artist_bbox(artist):
        if not artist.get_visible() or artist in {axs.patch, fig.patch}:
            return None
        if artist.__class__.__name__ in {'Spine', 'XAxis', 'YAxis'}:
            return None
        if artist.__class__.__name__ == 'Annotation' and artist.get_text() == '':
            getter_order = ('get_tightbbox',)
        else:
            getter_order = ('get_tightbbox', 'get_window_extent')

        for getter_name in getter_order:
            try:
                bbox = getattr(artist, getter_name)(renderer)
            except Exception:
                continue
            if bbox is not None and bbox.width > 0 and bbox.height > 0:
                 # Empty-text annotations can return a default tiny box at the origin.
                if bbox.x0 == 0 and bbox.y0 == 0 and bbox.width <= 1 and bbox.height <= 1:
                    continue
                return bbox
        return None

    artist_bboxes = [bbox for artist in axs.get_children() if (bbox := artist_bbox(artist)) is not None]

    if artist_bboxes:
        content_bbox = Bbox.union(artist_bboxes).transformed(fig.dpi_scale_trans.inverted())
    else:
        content_bbox = fig.get_tightbbox(renderer)
        assert content_bbox is not None
    content_bbox = content_bbox.padded(0.1)
    if name is not None:
        if is_seed:
            name = hashlib.sha256(name.encode()).hexdigest()
        #fig.savefig(name+'.'+format, format=format, bbox_inches=content_bbox)
        fig.savefig(name+'.'+format, format=format)
    plt.show()

class MVNgraph():
    """Class for plotting bending moment, shear force and normal force diagrams for beams."""
    def __init__(self) -> None:
        self.points = {}
        self.scale = 100

    def parabole(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> list:
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

    def interpolate_y(self, xs: list, ys: list, x_query: float) -> float:
        """interpolates the y-value for a given x-value based on the provided lists of x and y values.

        Args:
            xs (list): list of x-values corresponding to the y-values
            ys (list): list of y-values corresponding to the x-values
            x_query (float): The x-value for which to interpolate the y-value

        Returns:
            float: The interpolated y-value
        """
        for i in range(len(xs) - 1):
            if xs[i] <= x_query <= xs[i + 1]:
                return ys[i] + (ys[i + 1] - ys[i]) * ((x_query - xs[i]) / (xs[i + 1] - xs[i]))

    def rotate(self, beam: Beam, x: float, y: float) -> tuple:
        """Rotates a point (x, y) around the start point of a beam by the angle of the beam.

    Args:
        beam (Beam): The beam around which the point will be rotated.
        x (float): The x-coordinate of the point to be rotated, relative to the start point of the beam.
        y (float): The y-coordinate of the point to be rotated, relative to the start point of the beam.

    Returns:
        tuple[float, float]: The coordinates of the rotated point.
    """
        a = np.radians(beam.angle)

        return (beam.x1 + x*np.cos(a) - y*np.sin(a),
                beam.y1 + x*np.sin(a) + y*np.cos(a))

    def add_beam(self, beam: Beam, *pointtuples) -> None:
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

    def flip_sign(self, beam: Beam) -> None:
        self.points[beam]["flip_sign"] = True

    def find_crossings(self, x: list, y: list) -> list:
        """finds the x-values where the y-values cross zero or are zero, based on the provided lists of x and y values.

        Args:
            x (list): list of x-values corresponding to the y-values
            y (list): list of y-values corresponding to the x-values

        Returns:
            list: A list of x-values where the y-values cross zero or are zero
        """
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
    
    def plot(self) -> None:
        axs = plt.gca()
        axs.axis("equal")
        axs.axis('off')

        for beam in self.points:
            self.plot_beam(axs, beam)

    def plot_beam(self, axs, beam: Beam) -> None:
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


    def plot_labels(self, axs, beam: Beam, labels: list) -> None:
        # plot labels at characteristic points
        value, x_label, y_label = zip(*labels)
        y_label = [y_label_i*self.scale for y_label_i in y_label]
        x_label_rotated, y_label_rotated = zip(*(self.rotate(beam, xi, yi)
                                        for xi, yi in zip(x_label, y_label)))
        for i in range(len(x_label_rotated)):
            axs.annotate(value[i], (x_label_rotated[i], y_label_rotated[i]), ha='center',va='center', fontfamily='Times New Roman', fontsize=12)


class Mgraph(MVNgraph):
    """Class for plotting bending moment diagrams for beams."""
    def draw_signs(self, beam: Beam, crossings: list, x: list, y: list) -> None:
        """draws the bending moment signs (curved lines) on the plot using matplotlib

        Args:
            beam (Beam): The beam for which the bending moment diagram is being plotted.
            crossings (list): A list of x-values where the y-values cross zero or are zero.
            x (list): A list of x-values corresponding to the y-values.
            y (list): A list of y-values corresponding to the x-values.
        """
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
    def draw_signs(self, beam: Beam, crossings: list, x: list, y: list) -> None:
        """draws the shear force signs ('trappetjes') on the plot using matplotlib

        Args:
            beam (Beam): The beam for which the shear force diagram is being plotted.
            crossings (list): A list of x-values where the y-values cross zero or are zero.
            x (list): A list of x-values corresponding to the y-values.
            y (list): A list of y-values corresponding to the x-values.
        """
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
    def draw_signs(self, beam: Beam, crossings: list, x: list, y: list) -> None:
        """draws the normal force signs (+/-) on the plot using matplotlib

        Args:
            beam (Beam): The beam for which the normal force diagram is being plotted.
            crossings (list): A list of x-values where the y-values cross zero or are zero.
            x (list): A list of x-values corresponding to the y-values.
            y (list): A list of y-values corresponding to the x-values.
        """
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

# BUG Hinge in beam gaat niet altijd de goede kant op: wss verschilt het of het wel of niet het begin van de balk is


