import os

from modules.util import check


class ID_points:

    def __init__(self, nums=2):
        self.nums = nums
        self.value = 0

    def add(self):
        self.value = self.value + 1

        check = 0
        while True:
            check += 1
            if self.value < 10 ** (check - 1):
                break

            if ((self.value - (self.value % (10 ** (check - 1)))) / (10 ** (check - 1))) % 10 == self.nums:
                self.value += (10 - self.nums) * (10 ** (check - 1))


def generate_nod(points):
    if not os.path.exists(f'./{points}{points}network'):
        os.makedirs(f'./{points}{points}network')
    with open(f'./{points}{points}network/cross.nod.xml', "w") as routes:
        print(f"""<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">""", file=routes)
        print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                print(
                    f"""   <node id="{i + 1}{j + 1}" x="{1000.0 * j}" y="{-1000.0 * i}"  type="traffic_light"/>""", file=routes)
            print('\n', file=routes)

        for i in range(points):
            print(
                f"""   <node id="u{i + 1}" x="{1000.0 * i}" y="1000.0"  type="priority"/>""", file=routes)
        print('\n', file=routes)

        for i in range(points):
            print(
                f"""   <node id="d{i + 1}" x="{1000.0 * i}" y="{-1000.0 * points}"  type="priority"/>""", file=routes)
        print('\n', file=routes)

        for i in range(points):
            print(
                f"""   <node id="l{i + 1}" x="-1000.0" y="{-1000.0 * i}"  type="priority"/>""", file=routes)
        print('\n', file=routes)

        for i in range(points):
            print(
                f"""   <node id="r{i + 1}" x="{1000.0 * points}" y="{-1000.0 * i}"  type="priority"/>""", file=routes)
        print('\n', file=routes)
        print(f"""</nodes>""", file=routes)


def generate_edg(points):
    if not os.path.exists(f'./{points}{points}network'):
        os.makedirs(f'./{points}{points}network')
    with open(f'./{points}{points}network/cross.edg.xml', "w") as routes:
        print(f"""<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">""", file=routes)
        print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                if j == 0:
                    print(
                        f"""   <edge id="r{i + 1}{j + 1}" from="l{i+1}" to="{i+1}{j+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
                else:
                    print(
                        f"""   <edge id="r{i + 1}{j + 1}" from="{i+1}{j}" to="{i+1}{j+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print(
                f"""   <edge id="r{i + 1}{points + 1}" from="{i+1}{points}" to="r{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                if j == 0:
                    print(
                        f"""   <edge id="l{i + 1}{j}" to="l{i+1}" from="{i+1}{j+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
                else:
                    print(
                        f"""   <edge id="l{i + 1}{j}" to="{i+1}{j}" from="{i+1}{j+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print(
                f"""   <edge id="l{i + 1}{points}" to="{i+1}{points}" from="r{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                if j == 0:
                    print(
                        f"""   <edge id="d{j + 1}{i + 1}" from="u{i+1}" to="{j+1}{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
                else:
                    print(
                        f"""   <edge id="d{j + 1}{i + 1}" from="{j}{i+1}" to="{j+1}{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print(
                f"""   <edge id="d{points + 1}{i + 1}" from="{points}{i+1}" to="d{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                if j == 0:
                    print(
                        f"""   <edge id="u{j}{i + 1}" to="u{i+1}" from="{j+1}{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
                else:
                    print(
                        f"""   <edge id="u{j}{i + 1}" to="{j}{i+1}" from="{j+1}{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print(
                f"""   <edge id="u{points}{i + 1}" to="{points}{i+1}" from="d{i+1}" priority="1" numLanes="1" speed="15" />""", file=routes)
            print('\n', file=routes)
        print(f"""</edges>""", file=routes)


def generate_con(points):
    if not os.path.exists(f'./{points}{points}network'):
        os.makedirs(f'./{points}{points}network')
    with open(f'./{points}{points}network/cross.con.xml', "w") as routes:
        print(f"""<?xml version="1.0" encoding="UTF-8"?>
<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">""", file=routes)
        print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                print(
                    f"""   <connection from="r{i + 1}{j + 1}" to="r{i + 1}{j + 2}"/>""", file=routes)
            print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                print(
                    f"""   <connection to="l{i + 1}{j}" from="l{i + 1}{j + 1}"/>""", file=routes)
            print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                print(
                    f"""   <connection from="d{j + 1}{i + 1}" to="d{j + 2}{i + 1}"/>""", file=routes)
            print('\n', file=routes)

        for i in range(points):
            for j in range(points):
                print(
                    f"""   <connection to="u{j}{i + 1}" from="u{j + 1}{i + 1}"/>""", file=routes)
            print('\n', file=routes)

        print(f"""</connections>""", file=routes)


def generate_routefile(points, id, ns, we):
    if not os.path.exists(f'./{points}{points}network/route'):
        os.makedirs(f'./{points}{points}network/route')
    with open(f'./{points}{points}network/route/{id}.rou.xml', "w") as routes:
        print(f"""<routes>
    <vType accel="1.0" decel="5.0" id="ACar" length="2.0" maxSpeed="15.0" sigma="1.0" />""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            routes_str = f"d1{i + 1}"
            for j in range(points):
                routes_str += f" d{j + 2}{i + 1}"
            print(
                f"""    <route id="route_ns{i + 1}" edges="{routes_str}"/>""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            routes_str = f"u{points}{i + 1}"
            for j in range(points):
                routes_str += f" u{points - j - 1}{i + 1}"
            print(
                f"""    <route id="route_sn{i + 1}" edges="{routes_str}"/>""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            routes_str = f"r{i + 1}1"
            for j in range(points):
                routes_str += f" r{i + 1}{j + 2}"
            print(
                f"""    <route id="route_we{i + 1}" edges="{routes_str}"/>""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            routes_str = f"l{i + 1}{points}"
            for j in range(points):
                routes_str += f" l{i + 1}{points - j - 1}"
            print(
                f"""    <route id="route_ew{i + 1}" edges="{routes_str}"/>""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            print(
                f"""    <flow depart="1" id="flow_n_s{i + 1}_0_1800" route="route_ns{i + 1}" type="ACar" begin="0" end="1800" probability="{round((ns[i] * 2 + 2) / 10, 2)}" />""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            print(
                f"""    <flow depart="1" id="flow_s_n{i + 1}_0_1800" route="route_sn{i + 1}" type="ACar" begin="0" end="1800" probability="{round((ns[i] * 2 + 2) / 10, 2)}" />""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            print(
                f"""    <flow depart="1" id="flow_w_e{i + 1}_0_1800" route="route_we{i + 1}" type="ACar" begin="0" end="1800" probability="{round((we[i] * 2 + 2) / 10, 2)}" />""", file=routes)

        print('\n', file=routes)
        for i in range(points):
            print(
                f"""    <flow depart="1" id="flow_e_w{i + 1}_0_1800" route="route_ew{i + 1}" type="ACar" begin="0" end="1800" probability="{round((we[i] * 2 + 2) / 10, 2)}" />""", file=routes)

        print('\n', file=routes)
        print(f"""</routes>""", file=routes)


def generate_sumocfg(points, id):
    if not os.path.exists(f'./{points}{points}network/sumocfg'):
        os.makedirs(f'./{points}{points}network/sumocfg')

    with open(f'./{points}{points}network/sumocfg/{id}.sumocfg', "w") as sumocfg:
        print(f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
	<input>
		<net-file value="../cross.net.xml"/>
		<route-files value="../route/{id}.rou.xml"/>
	</input>
	<time>
		<begin value="0"/>
	</time>
	<report>
		<verbose value="true"/>
		<no-step-log value="true"/>
	</report>
</configuration>""", file=sumocfg)


def generate_basic(points):
    generate_nod(points)
    generate_edg(points)
    generate_con(points)
    os.system(
        f'netconvert --node-files=./{points}{points}network/cross.nod.xml --edge-files=./{points}{points}network/cross.edg.xml --connection-files=./{points}{points}network/cross.con.xml --output-file=./{points}{points}network/cross.net.xml')


def generate(points, max=1):
    #generate_routefile(points, 0)
    #generate_sumocfg(points, 0)

    id_check = ID_points()

    id_list, ns_list, we_list = [], [], []

    for _ in range(2 ** (2 * points)):
        id_str = str(id_check.value)
        if len(id_str) < (2 * points):
            for _ in range((2 * points) - len(id_str)):
                id_str = '0' + id_str

        ns_list.append([max * int(item) for item in id_str[:points]])
        we_list.append([max * int(item) for item in id_str[points:]])
        id_list.append(id_str)
        id_check.add()

        generate_sumocfg(points, id_str)
        generate_routefile(points, id_str, ns_list[-1], we_list[-1])


def generate_short(points, max=1):
    assert points == 2
    generate_sumocfg(points, '0011')
    generate_routefile(points, '0011', [0, 0], [max, max])

    generate_sumocfg(points, '1010')
    generate_routefile(points, '1010', [max, 0], [max, 0])

    generate_sumocfg(points, '0101')
    generate_routefile(points, '0101', [0, max], [0, max])

    generate_sumocfg(points, '1100')
    generate_routefile(points, '1100', [max, max], [0, 0])


if __name__ == '__main__':

    generate_basic(2)
    generate_basic(3)
    generate_basic(6)
