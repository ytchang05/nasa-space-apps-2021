import csv
import os
import random
import re
import time

import numpy as np
import scipy.spatial as sps
import vpython as vp


# Constants
ASTEROIDS_CSV: str = "./NASA_Asteroids.csv"
ASTEROIDS_PATH: str = "./NASA_Asteroids/"
GRAY: vp.vector = vp.vector(0.95, 0.95, 0.95)
SHAFT_WIDTH: float = 0.05


TABLE_CSS = """<style>
table, th, td {
  border: 0.5px solid black;
  border-collapse: collapse;
  padding: 0.5em;
  font-size: 99%;
}</style>"""


URL: str = '<a href="{0}">Link</a>'


# Generate asteroid info table
def gen_table(dat):
    return f"""{TABLE_CSS}<table>
    <tr>
        {''.join([f'<th>{header}</th>' for header in dat[0]])}
    </tr>
    <tr>
        {''.join([f'''<tr>{''.join([f"<td>{v if not v.startswith('https://') else URL.format(v)}</td>" for v in r])}</tr>''' for r in dat[1:]])}
    </tr>
</table>
    """


with open(ASTEROIDS_CSV, "r") as f:
    reader: csv.reader = csv.reader(f)
    data: list = [r for r in reader]


# Create selector canvas
SELECTOR: vp.canvas = vp.canvas(
    width=0,
    height=0,
    background=GRAY,
    align="none",
    resizable=False,
    autoscale=True,
    fov=vp.pi / 3,
    caption=gen_table(data),
    title="NASA Space Apps Challenge 2021: Team Starlight Beacon\n\n"
)


# Selector canvas placeholder sphere
vp.sphere(canvas=SELECTOR, color=GRAY, opacity=0, shininess=0)


# Asteroid selector menu
asteroid_menu = vp.menu(
    bind=lambda m: None,
    choices=["Select Asteroid"] + os.listdir(ASTEROIDS_PATH),
    pos=SELECTOR.title_anchor,
)


# Run button bound function
def run_button_bind(b: vp.button):

    # Make sure an option was selected
    if (not asteroid_menu.selected) or (asteroid_menu.selected == "Select Asteroid"):
        return

    print(f"Running: {asteroid_menu.selected}")

    # Disable button and selector menu
    b.disabled = True
    asteroid_menu.disabled = True
    b.visible = False
    asteroid_menu.visible = False
    SELECTOR.caption = []
    # Run the program
    run(asteroid_menu.selected)


# Run button
run_button: vp.button = vp.button(
    bind=run_button_bind, text="Run", pos=SELECTOR.title_anchor
)
mainm_button: vp.button = vp.button(
    bind=[], text="Asteroid & Its Projections", pos=SELECTOR.title_anchor
)


# Run the program
def run(asteroid: str):

    # Read shape (Wavefront .obj file)
    with open(os.path.join(ASTEROIDS_PATH, asteroid), "r") as f:
        shape: list = [re.split("/| +", line) for line in f.readlines()]

    # Main canvas
    canvas_main: vp.canvas = vp.canvas(
        title="",
        width=500,
        height=600,
        center=vp.vector(0, 0, 0),
        background=vp.color.black,
        align="left",
    )

    # Main canvas axes
    m_axis_rotation: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(1, 0, 0),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.white,
    )
    m_mark_rotation: vp.sphere = vp.sphere(
        pos=vp.vector(1, 0, 0),
        color=vp.color.white,
        make_trail=True, trail_type="points",
        interval=10, retain=80,
        trail_radius=0.01,
        radius=0.01,
    )
    m_axis_nutation: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(1.5, 0, 0),
        shaftwidth=SHAFT_WIDTH*0.7,
        color=vp.color.yellow,
    )
    m_axis_x: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(0, 0, -1),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.red,
    )
    m_axis_y: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(-1, 0, 0),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.green,
    )
    m_axis_z: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(0, -1, 0),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.blue,
    )

    # Find center of mass
    points = [
        list(map(float, line[1:]))
        for line in shape
        if (len(line) > 0) and (line[0] == "v")
    ]
    center_of_mass = np.sum(points, axis=0) / (len(points) - 1)

    # Asteroid shape vertices
    vertices: list = [vp.vertex(pos=vp.vector(0, 0, 0))]
    vertices_xy: list = [vp.vertex(pos=vp.vector(0, 0, 0))]
    vertices_yz: list = [vp.vertex(pos=vp.vector(0, 0, 0))]
    vertices_zx: list = [vp.vertex(pos=vp.vector(0, 0, 0))]
    for line in shape:
        if (len(line) > 0) and (line[0] == "v"):
            for v_list in (vertices, vertices_xy, vertices_yz, vertices_zx):
                v_list.append(
                    vp.vertex(
                        pos=vp.vector(
                            *(
                                (
                                    np.array([float(p) for p in line[1:]])
                                    - center_of_mass
                                ).tolist()
                            )
                        ),
                        color = vp.color.gray((random.randint(0,20) + 20) / 60)
                    )
                )

    # Asteroid shape faces
    faces: list = []
    faces_xy: list = []
    faces_yz: list = []
    faces_zx: list = []
    face_vertices: list = []
    face_vertices_already_in_list: np.array = np.full(len(vertices), False)
    for line in shape:
        if (len(line) > 0) and (line[0] == "f"):
            x = [line[1]] + (
                [line[3], line[5]] if (len(line) > 4) else [line[2], line[3]]
            )
            for f_list, v_list in (
                (faces, vertices),
                (faces_xy, vertices_xy),
                (faces_yz, vertices_yz),
                (faces_zx, vertices_zx),
            ):
                f_list.append(
                    vp.triangle(
                        v0=v_list[int(x[0])],
                        v1=v_list[int(x[1])],
                        v2=v_list[int(x[2])],
                    )
                )
            for i in range(3):
                if not face_vertices_already_in_list[int(x[i])]:
                    face_vertices.append(getattr(faces[-1], f"v{i}"))
                    face_vertices_already_in_list[int(x[i])] = True

    # Separator canvas
    canvas_sep1: vp.canvas = vp.canvas(
        title="",
        width=10,
        height=600,
        center=vp.vector(0, 0, 0),
        background=vp.color.white,
        align="left",
        caption="",
    )
    vp.sphere(radius=0.001)
    canvas_sep1.camera.axis = vp.vector(100,100,100)

    # Rotation control canvas
    canvas_ctrl: vp.canvas = vp.canvas(
        title="",
        width=350,
        height=300,
        center=vp.vector(0, 0, 0),
        background=vp.color.black,
        align="left",
    )
#   canvas_ctrl.caption="Adjust Rotational Axis"
    # title button
    canvas_ctrl_button: vp.button = vp.button(
        bind=[],
        width=200,
#        text="Adjust Rotational Axis",
        text="Adjust Rotational Axis                                                    Adjust Nutation                                                       ",
        pos=canvas_ctrl.title_anchor
    )

    # Rotation control canvas axes
    c_axis_x: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(0, 0, -1),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.red,
    )
    c_axis_y: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(0, -1, 0),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.green,
    )
    c_axis_z: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(-1, 0, 0),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.blue,
    )

    # Separator canvas
    canvas_sep2: vp.canvas = vp.canvas(
        title="",
        width=10,
        height=300,
        center=vp.vector(0, 0, 0),
        background=vp.color.white,
        align="left",
        caption="",
    )
    vp.sphere(radius=0.001)
    canvas_sep2.camera.axis = vp.vector(100,100,100)

    # Nutation control canvas
    canvas_nut: vp.canvas = vp.canvas(
        title="",
        width=350,
        height=300,
        center=vp.vector(0, 0, 0),
        background=vp.color.black,
        align="left",
    )
    #   canvas_ctrl.caption="Adjust Rotational Axis"

    # Nutation control canvas axes
    n_axis_x: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(0, 0, -1),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.red,
    )
    n_axis_y: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(0, -1, 0),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.green,
    )
    n_axis_z: vp.arrow = vp.arrow(
        pos=vp.vector(0, 0, 0),
        axis=vp.vector(-1, 0, 0),
        shaftwidth=SHAFT_WIDTH,
        color=vp.color.blue,
    )


    # Light curve graphs
    vp.graph(
        scroll=True, fast=False, xmin=0, xmax=10, align="left", width=700, height=300,
        title="Light curves observed from three mutually perpendicular planes"
    )
    graph_xy: vp.gcurve = vp.gcurve(color=vp.color.red)
    graph_yz: vp.gcurve = vp.gcurve(color=vp.color.green)
    graph_zx: vp.gcurve = vp.gcurve(color=vp.color.blue)

    # Rotate!
    t: float = 0
    while True:

        # Rate to 30fps
        vp.rate(30)

        # Change nutational axis to match control
        m_axis_nutation.axis = vp.vector(
            -1 * vp.dot(canvas_nut.forward, vp.vector(1, 0, 0)),
            vp.dot(canvas_nut.forward, vp.vector(0, 1, 0)),
            vp.dot(canvas_nut.forward, vp.vector(0, 0, 1)),
        )
        # Change rotational axis to match control
        m_axis_rotation.axis = vp.vector(
            -1 * vp.dot(canvas_ctrl.forward, vp.vector(1, 0, 0)),
            vp.dot(canvas_ctrl.forward, vp.vector(0, 1, 0)),
            vp.dot(canvas_ctrl.forward, vp.vector(0, 0, 1)),
        )

        # Rotate with Rodrigues' rotation formula
        def rotate(vec, k, theta):  # vector, spin_axis
            return (
                vec * vp.cos(theta)
                + vp.cross(k, vec) * vp.sin(theta)
                + k * vp.dot(k, vec) * (1 - vp.cos(theta))
            )

        dircos = [vp.dot(m_axis_nutation.axis, vp.vector(1, 0, 0)),
                vp.dot(m_axis_nutation.axis, vp.vector(0, 1, 0)),
                vp.dot(m_axis_nutation.axis, vp.vector(0, 0, 1))]
        dirvec = [vp.vector(1,0,0), vp.vector(0,1,0), vp.vector(0,0,1) ]
        m_axis_nutation_p = vp.cross(m_axis_nutation.axis, dirvec[ dircos.index(max(dircos)) ])
        m_axis_rotation.axis = rotate(m_axis_rotation.axis,
                                      m_axis_nutation_p,
                                      (vp.dot(canvas_nut.forward, vp.vector(1, 0, 0))) )
        m_axis_rotation.axis = rotate(m_axis_rotation.axis,
                                      m_axis_nutation.axis,
                                      2*vp.pi*(((10*t)%360)/360 ))
        m_mark_rotation.pos = m_axis_rotation.axis
        # Rotate each vertex
#        m_axis_rotation.axis = rotate(m_axis_rotation.axis, normalized, vp.pi / 20)
        normalized: vp.vector = m_axis_rotation.axis.norm()

        for fv in face_vertices:
            fv.pos = rotate(fv.pos, normalized, vp.pi / 20)

        time.sleep(0.1)

        # Create projections
        projection_xy: list = []
        projection_yz: list = []
        projection_zx: list = []
        for i, f in enumerate(faces):
            for v in (f.v0, f.v1, f.v2):
                projection_xy.append([v.pos.x, v.pos.y])
                projection_yz.append([v.pos.y, v.pos.z])
                projection_zx.append([v.pos.z, v.pos.x])
            for j in ("v0", "v1", "v2"):
                vtx: vp.vertex = getattr(f, j)
                getattr(faces_xy[i], j).pos = vp.vector(vtx.pos.x, vtx.pos.y, -2)
                getattr(faces_yz[i], j).pos = vp.vector(-2, vtx.pos.y, vtx.pos.z)
                getattr(faces_zx[i], j).pos = vp.vector(vtx.pos.x, -2, vtx.pos.z)

        # Use ConvexHull to find area (0th order approximation of light curve)
        hull_xy: sps.ConvexHull = sps.ConvexHull(projection_xy)
        hull_yz: sps.ConvexHull = sps.ConvexHull(projection_yz)
        hull_zx: sps.ConvexHull = sps.ConvexHull(projection_zx)

        # Graph area
        graph_xy.plot(t, hull_xy.area)
        graph_yz.plot(t, hull_yz.area)
        graph_zx.plot(t, hull_zx.area)

        t += 0.1
        time.sleep(0.03)
