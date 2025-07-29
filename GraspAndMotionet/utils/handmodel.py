import json
import os
import pytorch_kinematics as pk
import torch.nn
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
from utils.rot6d import *
import trimesh.sample
import pytorch3d
from pytorch3d.ops import knn_points
class HandModel:
    def __init__(self, robot_name, urdf_filename, mesh_path,
                 batch_size=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 hand_scale=2.
                 ):
        self.device = device
        self.robot_name = robot_name
        self.batch_size = batch_size
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)
        # prepare contact point basis and surface point samples
        # self.no_contact_dict = json.load(open(os.path.join('data', 'urdf', 'intersection_%s.json'%robot_name)))
        self.dis_key_point = {
        "palm": [],
        "ffproximal": [[-0.0002376327756792307, -0.009996689856052399, 0.038666076958179474], [-0.0035445429384708405, -0.009337972849607468, 4.728326530312188e-05], [0.0042730518616735935, -0.0090293288230896, 0.018686404451727867], [-0.003900623880326748, -0.009198302403092384, 0.027359312400221825], [-0.0034948040265589952, -0.009357482194900513, 0.011004473082721233], [0.004485304467380047, -0.008933592587709427, 0.005608899053186178], [0.00421907939016819, -0.009053671732544899, 0.030992764979600906], [-0.003979427739977837, -0.009167392738163471, 0.01910199038684368], [-0.0037133553996682167, -0.009271756745874882, 0.04499374330043793], [0.0034797703847289085, -0.009367110207676888, 0.044556595385074615]],
        "ffmiddle": [[-0.0019831678364425898, -0.007794334553182125, 0.009099956601858139], [0.0017110002227127552, -0.007856125012040138, 0.024990297853946686], [0.003553177695721388, -0.007216729689389467, 0.0004225552547723055], [0.003431637305766344, -0.007271687965840101, 0.01607479713857174], [-0.0025954796001315117, -0.007619044743478298, 0.0195100586861372], [-0.0028260457329452038, -0.007528608664870262, 0.0024113801773637533], [0.003367392346262932, -0.007300738710910082, 0.0064179981127381325], [-0.0014348605182021856, -0.007911290973424911, 0.014330973848700523], [0.0024239378981292248, -0.0076669249683618546, 0.011192334815859795], [-0.003152574645355344, -0.007400532253086567, 0.02429058402776718]],
        "ffdistal": [[-0.00094795529730618, -0.006982842925935984, 0.01811189576983452], [0.002626439556479454, -0.006539319641888142, 4.996722418582067e-05], [-0.0034360431600362062, -0.00615164078772068, 0.008426538668572903], [0.002973517868667841, -0.006382592022418976, 0.011918909847736359], [0.0026527668815106153, -0.006527431774884462, 0.02366715297102928], [-0.0034155123867094517, -0.006162168458104134, 0.0019269119948148727], [-0.0033331606537103653, -0.006204396951943636, 0.023483257740736008], [0.0025241682305932045, -0.0065797604620456696, 0.005907486192882061], [-0.00349381472915411, -0.006122016813606024, 0.0137711763381958], [0.0031557006295770407, -0.006300325505435467, 0.01670246012508869]],
        "mfproximal": [[-0.0020669603254646063, -0.009777018800377846, 0.006492570973932743], [0.00465710973367095, -0.00884400587528944, 0.04495971277356148], [-0.004878615960478783, -0.008722265250980854, 0.027143988758325577], [0.005264972802251577, -0.008492529392242432, 0.017509466037154198], [0.005084634758532047, -0.008596803992986679, 0.032350800931453705], [-0.004348081536591053, -0.008994411677122116, 0.0383140966296196], [-0.004989673383533955, -0.0086652971804142, 0.01689181476831436], [0.005276953335851431, -0.008485602214932442, 0.0004979652003385127], [0.0050704991444945335, -0.008604977279901505, 0.009669311344623566], [0.0027988858055323362, -0.00959551241248846, 0.024868451058864594]],
        "mfmiddle": [[-0.00042295613093301654, -0.008031148463487625, 0.011103776283562183], [0.003634985536336899, -0.007179737091064453, 0.02486497163772583], [0.0035494803451001644, -0.007218401413410902, 0.00027021521236747503], [-0.003957465291023254, -0.007006912492215633, 0.019527770578861237], [-0.003908041398972273, -0.007031631655991077, 0.003815919626504183], [0.0035529686138033867, -0.007216824218630791, 0.017309214919805527], [0.003624177537858486, -0.007184624206274748, 0.006498508155345917], [-0.0026851133443415165, -0.007583887316286564, 0.024684462696313858], [-0.003890471300110221, -0.007041062694042921, 0.014749204739928246], [0.00044322473695501685, -0.008030962198972702, 0.02091158926486969]],
        "mfdistal": [[0.0002320836065337062, -0.007037499453872442, 0.023355133831501007], [0.0036455930676311255, -0.006024540401995182, 6.64829567540437e-05], [-0.0032930555753409863, -0.006224961951375008, 0.010883713141083717], [0.004127402324229479, -0.005703427363187075, 0.015460449270904064], [-0.003456553677096963, -0.006141123361885548, 0.002992440015077591], [0.003985205665230751, -0.005805530119687319, 0.00806250236928463], [-0.0035007710102945566, -0.0061184498481452465, 0.017718486487865448], [0.004287987481802702, -0.00558812078088522, 0.02120518684387207], [0.0011520618572831154, -0.006951729767024517, 0.004545787815004587], [0.0013482635840773582, -0.006914287339895964, 0.011958128772675991]],
        "rfproximal": [[-0.002791694598272443, -0.009589731693267822, 0.015829697251319885], [0.003399983746930957, -0.009393873624503613, 0.04477155953645706], [0.003595913527533412, -0.009328149259090424, 0.0003250864101573825], [-0.0048502604477107525, -0.008736810646951199, 0.031506575644016266], [0.004417791962623596, -0.008964043110609055, 0.024784449487924576], [-0.004430862609297037, -0.008951948024332523, 0.006260100286453962], [0.004398377146571875, -0.008972801268100739, 0.03525833785533905], [0.00454053096473217, -0.008908682502806187, 0.009322012774646282], [-0.00461477879434824, -0.008857605047523975, 0.04111073166131973], [-0.00443872157484293, -0.008947916328907013, 0.023684965446591377]],
        "rfmiddle": [[-0.0024211457930505276, -0.007671383209526539, 0.003983666189014912], [0.002596878679469228, -0.007608974818140268, 0.024917811155319214], [-0.003061287570744753, -0.007436338346451521, 0.015137670561671257], [0.0027961833402514458, -0.007542191073298454, 0.009979978203773499], [0.0029241566080600023, -0.007499308791011572, 0.01832345686852932], [0.0027976972050964832, -0.007541683502495289, 6.793846841901541e-05], [-0.0028554939199239016, -0.007517057936638594, 0.021669354289770126], [-0.00278903404250741, -0.007543126121163368, 0.009370749816298485], [0.002888968912884593, -0.007511099800467491, 0.005062070209532976], [0.0010934327729046345, -0.007965755648911, 0.013925164006650448]],
        "rfdistal": [[0.004119039047509432, -0.005709432996809483, 0.022854819893836975], [-0.004941829480230808, -0.005017726682126522, 2.3880027583800256e-05], [0.005020809359848499, -0.0049394648522138596, 0.009076559916138649], [-0.0051115998066961765, -0.0048520066775381565, 0.014685478061437607], [0.004567775409668684, -0.00535866804420948, 2.354436037421692e-05], [-0.004747811239212751, -0.005207117181271315, 0.023783991113305092], [-0.0021952472161501646, -0.006697545759379864, 0.006976036354899406], [0.0026042358949780464, -0.006549346260726452, 0.015848159790039062], [-0.0018935244297608733, -0.0067823501303792, 0.019202016294002533], [-0.00021895798272453249, -0.007044903934001923, 0.0015841316198930144]],
        "lfmetacarpal": [],
        "lfproximal": [[-0.001103847287595272, -0.009932574816048145, 0.023190606385469437], [0.004271919839084148, -0.009029388427734375, 0.00020035798661410809], [0.0033563950564712286, -0.009408160112798214, 0.04472788795828819], [-0.00359934801235795, -0.009316868148744106, 0.010634070262312889], [-0.0028604455292224884, -0.009570563212037086, 0.0347319096326828], [0.004299539607018232, -0.009016930125653744, 0.01569746620953083], [-0.0040543111972510815, -0.009138413704931736, 0.0022569862194359303], [0.004445703700184822, -0.008951003663241863, 0.03112208843231201], [0.003617372363805771, -0.009320616722106934, 0.007815317250788212], [-0.003905154298990965, -0.009196918457746506, 0.04234839603304863]],
        "lfmiddle": [[-0.0007557488279417157, -0.00800648145377636, 0.006158251781016588], [0.003424291731789708, -0.007274557836353779, 0.024703215807676315], [-0.004018992651253939, -0.006975765340030193, 0.01652413047850132], [0.003509017638862133, -0.007236245553940535, 0.013466663658618927], [-0.0038151403423398733, -0.007080549374222755, 0.023723633959889412], [0.0034917076118290424, -0.007244073320180178, 0.0006077417056076229], [-0.003889185143634677, -0.0070418380200862885, 0.00041095237247645855], [0.0013372180983424187, -0.007935309782624245, 0.019177529960870743], [-0.0038003893569111824, -0.0070875901728868484, 0.010691785253584385], [0.0034656336065381765, -0.007255863398313522, 0.008516711182892323]],
        "lfdistal": [[0.0014511797344312072, -0.006890428718179464, 0.004574548453092575], [-0.0024943388998508453, -0.006586590316146612, 0.023863397538661957], [0.0029716803692281246, -0.006382519379258156, 0.014716518111526966], [-0.002874345052987337, -0.006437260191887617, 0.010602368041872978], [-0.0028133057057857513, -0.006461246870458126, 0.00012360091204755008], [0.00285350508056581, -0.006435882765799761, 0.020840927958488464], [-0.002667121822014451, -0.006518691778182983, 0.017100241035223007], [0.0029705220367759466, -0.006383041851222515, 0.009500452317297459], [0.002596389502286911, -0.006551986560225487, 0.00017241919704247266], [-0.002902866108343005, -0.006426053121685982, 0.0050438339821994305]],
        "thbase": [],
        "thproximal": [],
        "thhub": [],
        "thmiddle": [[-0.010736164636909962, -0.0023433465976268053, 0.005364177282899618], [-0.009576565586030483, 0.005389457568526268, 0.031773995608091354], [-0.010324702598154545, -0.0037935571745038033, 0.020191052928566933], [-0.009223436936736107, 0.005977252032607794, 0.0133969122543931], [-0.008859770372509956, 0.006510394625365734, 0.0007552475435659289], [-0.010023333132266998, -0.004508777987211943, 0.03042592667043209], [-0.009238336235284805, 0.005955408792942762, 0.022636638954281807], [-0.010435031726956367, -0.0034471736289560795, 0.012765333987772465], [-0.010987512767314911, 0.0005528760375455022, 0.026306135579943657], [-0.010371722280979156, 0.0036525875329971313, 0.007199263200163841]],
        "thdistal": [[-0.00094795529730618, -0.006982842925935984, 0.01811189576983452], [0.002626439556479454, -0.006539319641888142, 4.996722418582067e-05], [-0.0034360431600362062, -0.00615164078772068, 0.008426538668572903], [0.002973517868667841, -0.006382592022418976, 0.011918909847736359], [0.0026527668815106153, -0.006527431774884462, 0.02366715297102928], [-0.0034155123867094517, -0.006162168458104134, 0.0019269119948148727], [-0.0033331606537103653, -0.006204396951943636, 0.023483257740736008], [0.0025241682305932045, -0.0065797604620456696, 0.005907486192882061], [-0.00349381472915411, -0.006122016813606024, 0.0137711763381958], [0.0031557006295770407, -0.006300325505435467, 0.01670246012508869]]
        }
        self.keypoints = {
            "forearm": [],
            "wrist": [],
            "palm": [],
            "ffknuckle": [],
            "ffproximal": [[0, 0, 0.024]],
            "ffmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "ffdistal": [[0, 0, 0.024]],
            "fftip": [],
            "mfknuckle": [],
            "mfproximal": [[0, 0, 0.024]], 
            "mfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "mfdistal": [[0, 0, 0.024]],
            "mftip":[],
            "rfknuckle": [],
            "rfproximal": [[0, 0, 0.024]], 
            "rfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "rfdistal": [[0, 0, 0.024]],
            "lfmetacarpal": [],
            "lfknuckle": [],
            "lfproximal": [[0, 0, 0.024]],
            "lfmiddle": [[0, 0, 0], [0, 0, 0.025]],
            "lfdistal": [[0, 0, 0.024]],
            "lftip": [],
            "thbase": [], 
            "thproximal": [[0, 0, 0.038]], 
            "thhub": [],
            "thmiddle": [[0, 0, 0.032]], 
            "thdistal": [[0, 0, 0.026]],
            "thtip":[]
        }
        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        # prepare contact point basis and surface point samples
        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}

        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []
        verts_bias = 0

        if robot_name == 'shadowhand':
            self.palm_toward = torch.tensor([0., -1., 0., 0.], device=self.device).reshape(1, 1, 4).repeat(self.batch_size, 1, 1)
        else:
            raise NotImplementedError

        for i_link, link in enumerate(visual.links):
            # print(f"Processing link #{i_link}: {link.name}")
            # load mesh
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                # print(link.visuals[0])
                if robot_name == 'shadowhand' or robot_name == 'allegro' or robot_name == 'barrett':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                elif robot_name == 'allegro':
                    filename = f"{link.visuals[0].geometry.filename.split('/')[-2]}/{link.visuals[0].geometry.filename.split('/')[-1]}"
                else:
                    filename = link.visuals[0].geometry.filename
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(
                    radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                # print('---')
                # print(link.visuals[0].origin.rpy, rotation)
                # print('---')
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])

            # Surface point
            # mesh.sample(int(mesh.area * 100000)) * scale
            if self.robot_name == 'shadowhand':
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=64)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
            else:
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=128)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)

            if self.robot_name == 'barrett':
                if link.name in ['bh_base_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'ezgripper':
                if link.name in ['left_ezgripper_palm_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[1., 0., 0.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'robotiq_3finger':
                if link.name in ['gripper_palm']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)

            pts *= scale
            # pts = mesh.sample(128) * scale
            # print(link.name, len(pts))
            # new
            if robot_name == 'shadowhand':
                pts = pts[:, [0, 2, 1]]
                pts_normal = pts_normal[:, [0, 2, 1]]
                pts[:, 1] *= -1
                pts_normal[:, 1] *= -1

            pts = np.matmul(rotation, pts.T).T + translation
            # pts_normal = np.matmul(rotation, pts_normal.T).T
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
            self.surface_points[link.name] = torch.from_numpy(pts).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            if robot_name == 'shadowhand':
                self.mesh_verts[link.name] = self.mesh_verts[link.name][:, [0, 2, 1]]
                self.mesh_verts[link.name][:, 1] *= -1
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)
            link_vertices = torch.tensor(self.mesh_verts[link.name], dtype=torch.float)
            link_faces = torch.tensor(self.mesh_faces[link.name], dtype=torch.long)

        # new 2.1
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute':
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.revolute_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_lower = torch.Tensor(
            self.revolute_joints_q_lower).repeat([self.batch_size, 1]).to(device)
        self.revolute_joints_q_upper = torch.Tensor(
            self.revolute_joints_q_upper).repeat([self.batch_size, 1]).to(device)

        self.current_status = None

        self.scale = hand_scale

    def update_kinematics(self, q):
        if q.shape[1] == 3+24:
            self.global_translation = q[:, :3]
            self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(torch.tensor([1., 0., 0., 0., 1., 0.], device='cuda').view(1, 6).repeat(q.shape[0], 1))
            self.current_status = self.robot.forward_kinematics(q[:, 3:])
        elif q.shape[1] == 3+22:#3+22
            self.global_translation = q[:, :3]
            self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(torch.tensor([1., 0., 0., 0., 1., 0.], device='cuda').view(1, 6).repeat(q.shape[0], 1))
            expanded_joint_data = torch.cat([torch.zeros((q.shape[0], 2), device=q.device), q[:, 3:]], dim=1)#00+22
            self.current_status = self.robot.forward_kinematics(expanded_joint_data)

        elif q.shape[1] == 3+6+22:#3+6+22
            self.global_translation = q[:, :3]
            self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
            expanded_joint_data = torch.cat([torch.zeros((q.shape[0], 2), device=q.device), q[:, 9:]], dim=1)#00+22
            self.current_status = self.robot.forward_kinematics(expanded_joint_data)

        else:# 3+6+24
            self.global_translation = q[:, :3]
            self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
            self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def save_point_cloud(self, points, filename):
        point_cloud = trimesh.points.PointCloud(points)
        point_cloud.export(filename)
        print(f"Saved point cloud to {filename}")

    def save_mesh(self, face_verts, filename):
        vertices = face_verts.reshape(-1, 3)
        
        faces = np.arange(vertices.shape[0]).reshape(-1, 3)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(filename)
        print(f"Saved mesh to {filename}")


    
    def get_keypoints(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        keypoints = [
            self.current_status[link_name].transform_points(torch.tensor(self.keypoints[link_name], device=self.device, dtype=torch.float32)).expand(self.batch_size, -1, -1)
            for link_name in self.keypoints if len(self.keypoints[link_name]) > 0]
        keypoints = torch.cat(keypoints, dim=1)
        keypoints = torch.bmm(keypoints, self.global_rotation.transpose(1, 2)) + self.global_translation.unsqueeze(1)
        return keypoints* self.scale
    
    def get_dis_keypoints(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        dis_points = [
            self.current_status[link_name].transform_points(torch.tensor(self.dis_key_point[link_name], device=self.device, dtype=torch.float32)).expand(self.batch_size, -1, -1)
            for link_name in self.dis_key_point if len(self.dis_key_point[link_name]) > 0]
        dis_points = torch.cat(dis_points, dim=1)
        dis_points = torch.bmm(dis_points, self.global_rotation.transpose(1, 2)) + self.global_translation.unsqueeze(1)
        return dis_points* self.scale
    

    def get_surface_points(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            if link_name  in ['forearm']:
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation.float(), surface_points.transpose(1, 2)).transpose(1,
                                                                                                      2) + self.global_translation.unsqueeze(
            1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_palm_points(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in ['palm']:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale

    def get_palm_toward_point(self, q=None):
        if q is not None:
            self.update_kinematics(q)

        link_name = 'palm'
        trans_matrix = self.current_status[link_name].get_matrix()
        palm_toward_point = torch.matmul(trans_matrix, self.palm_toward.transpose(1, 2)).transpose(1, 2)[..., :3]
        palm_toward_point = torch.matmul(self.global_rotation, palm_toward_point.transpose(1, 2)).transpose(1, 2)

        return palm_toward_point.squeeze(1)

    def get_palm_center_and_toward(self, q=None):
        if q is not None:
            self.update_kinematics(q)

        palm_surface_points = self.get_palm_points()
        palm_toward_point = self.get_palm_toward_point()

        palm_center_point = torch.mean(palm_surface_points, dim=1, keepdim=False)
        return palm_center_point, palm_toward_point

    def get_surface_points_and_normals(self, q=None):
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
            surface_normals.append(
                torch.matmul(trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[...,
                :3])
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1,
                                                                                                      2) + self.global_translation.unsqueeze(
            1)
        surface_normals = torch.matmul(self.global_rotation, surface_normals.transpose(1, 2)).transpose(1, 2)

        return surface_points * self.scale, surface_normals

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            if link_name  in ['forearm']:
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            if link_name  in ['forearm']:
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(x=transformed_v[:, 0], y=transformed_v[:, 1], z=transformed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
        return data


def get_handmodel(batch_size, device, hand_scale=1., robot='shadowhand'):
    urdf_assets_meta = json.load(open("assets/shadowhand_with_forearm_assets/urdf/urdf_assets_meta.json"))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    hand_model = HandModel(robot, urdf_path, meshes_path, batch_size=batch_size, device=device, hand_scale=hand_scale)
    return hand_model


def compute_collision(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    :param obj_pcd_nor: N_obj x 6
    :param hand_surface_points: B x N_hand x 3
    :return:
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[0]
    n_hand = hand_pcd.shape[1]

    obj_pcd = obj_pcd_nor[:, :3]
    obj_nor = obj_pcd_nor[:, 3:6]

    # batch the obj pcd
    batch_obj_pcd = obj_pcd.unsqueeze(0).repeat(b, 1, 1).view(b, 1, n_obj, 3)
    batch_obj_pcd = batch_obj_pcd.repeat(1, n_hand, 1, 1)
    # batch the hand pcd
    batch_hand_pcd = hand_pcd.view(b, n_hand, 1, 3).repeat(1, 1, n_obj, 1)
    # compute the pair wise dist
    hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
    hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
    # gather the obj points and normals w.r.t. hand points
    hand_obj_points = torch.stack([obj_pcd[x, :] for x in hand_obj_indices], dim=0)
    hand_obj_normals = torch.stack([obj_nor[x, :] for x in hand_obj_indices], dim=0)
    # compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # signs dot dist to compute collision value
    collision_value = (hand_obj_signs * hand_obj_dist).max(dim=1).values
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    return collision_value

def compute_collision_filter(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    :param obj_pcd_nor: N_obj x 6
    :param hand_surface_points: B x N_hand x 3
    :return:
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[0]
    n_hand = hand_pcd.shape[1]

    obj_pcd = obj_pcd_nor[:, :3]
    obj_nor = obj_pcd_nor[:, 3:6]

    # batch the obj pcd
    batch_obj_pcd = obj_pcd.unsqueeze(0).repeat(b, 1, 1).view(b, 1, n_obj, 3)
    batch_obj_pcd = batch_obj_pcd.repeat(1, n_hand, 1, 1)
    # batch the hand pcd
    batch_hand_pcd = hand_pcd.view(b, n_hand, 1, 3).repeat(1, 1, n_obj, 1)
    # compute the pair wise dist
    hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
    hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
    # gather the obj points and normals w.r.t. hand points
    hand_obj_points = torch.stack([obj_pcd[x, :] for x in hand_obj_indices], dim=0)
    hand_obj_normals = torch.stack([obj_nor[x, :] for x in hand_obj_indices], dim=0)
    # compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # signs dot dist to compute collision value
    collision_value = (hand_obj_signs * hand_obj_dist).max(dim=1).values
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    print(collision_value)
    return collision_value
def ERF_loss(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    Calculate the penalty loss based on point cloud and normal.

    :param obj_pcd_nor: B x N_obj x 6 (object point cloud with normals)
    :param hand_pcd: B x N_hand x 3 (hand point cloud)
    :return: ERF_loss (scalar)
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[1]
    n_hand = hand_pcd.shape[1]

    # Separate object point cloud and normals
    obj_pcd = obj_pcd_nor[:, :, :3]
    obj_nor = obj_pcd_nor[:, :, 3:6]

    # Compute K-nearest neighbors
    knn_result = knn_points(hand_pcd, obj_pcd, K=1, return_nn=True)
    distances = knn_result.dists
    indices = knn_result.idx
    knn = knn_result.knn
    distances = distances.sqrt()
    # Extract the closest object points and normals
    hand_obj_points = torch.gather(obj_pcd, 1, indices.expand(-1, -1, 3))
    hand_obj_normals = torch.gather(obj_nor, 1, indices.expand(-1, -1, 3))
    # Compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    hand_obj_signs = (hand_obj_signs > 0.).float()
    # Compute collision value
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    collision_value = (hand_obj_signs * distances.squeeze(2)).max(dim=1).values
    ERF_loss = collision_value.mean()
    return ERF_loss

def SPF_loss(dis_points, obj_pcd: torch.Tensor, thres_dis = 0.02 ):
    dis_points = dis_points.to(dtype=torch.float32)
    obj_pcd = obj_pcd.to(dtype=torch.float32)
    dis_pred = pytorch3d.ops.knn_points(dis_points, obj_pcd).dists[:, :, 0] # 64*140  # squared chamfer distance from object_pc to contact_candidates_pred
    small_dis_pred = dis_pred < thres_dis ** 2# 64*140
    SPF_loss = dis_pred[small_dis_pred].sqrt().sum() / (small_dis_pred.sum().item() + 1e-5)#1
    return SPF_loss

def SRF_loss(points):
    B, *points_shape = points.shape
    dis_spen = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
    dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
    dis_spen = 0.02 - dis_spen
    dis_spen[dis_spen < 0] = 0
    SPF_loss = dis_spen.sum() / B
    return SPF_loss

