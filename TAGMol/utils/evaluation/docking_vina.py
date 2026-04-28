import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import os

from utils.reconstruct import reconstruct_from_generated
from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask

VINA_EXE = '/usr/bin/vina'


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format != 'sdf':
            raise ValueError(f'mol_format {mol_format} not supported')
        self.input_mol = input_mol

    def get_pdbqt(self, lig_pdbqt=None):
        if lig_pdbqt is None:
            raise ValueError('lig_pdbqt must be provided')
        cmd = [
            'obabel',
            self.input_mol,
            '-O', lig_pdbqt,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return lig_pdbqt
        

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        return prot_pqr

    def get_pdbqt(self, prot_pdbqt):
        subprocess.run(
            [
                'obabel',
                self.prot,
                '-O', prot_pdbqt,
                '-xr',
            ],
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def _run_vina(self, mode, exhaustiveness):
        out_path = tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False)
        out_path.close()
        cmd = [
            VINA_EXE,
            '--receptor', self.prot_pdbqt,
            '--ligand', self.lig_pdbqt,
            '--center_x', str(self.pocket_center[0]),
            '--center_y', str(self.pocket_center[1]),
            '--center_z', str(self.pocket_center[2]),
            '--size_x', str(self.box_size[0]),
            '--size_y', str(self.box_size[1]),
            '--size_z', str(self.box_size[2]),
            '--cpu', '1',
            '--verbosity', '0',
        ]
        if mode == 'score_only':
            cmd.append('--score_only')
        elif mode == 'minimize':
            cmd.append('--local_only')
        elif mode == 'dock':
            cmd.extend(['--exhaustiveness', str(exhaustiveness), '--out', out_path.name])
        else:
            raise ValueError
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = proc.stdout
        score = None
        for line in stdout.splitlines():
            if mode == 'dock':
                stripped = line.strip()
                if stripped.startswith('1 '):
                    parts = stripped.split()
                    score = float(parts[1])
                    break
            elif 'Estimated Free Energy of Binding' in line:
                score = float(line.split(':')[-1].split()[0])
                break
        pose = None
        if mode == 'dock' and os.path.exists(out_path.name):
            with open(out_path.name, 'r', encoding='utf-8') as f:
                pose = f.read()
            os.remove(out_path.name)
        else:
            if os.path.exists(out_path.name):
                os.remove(out_path.name)
        if score is None:
            raise ValueError(f'Failed to parse vina output for mode {mode}: {stdout}')
        return score, pose

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        if not os.path.exists(VINA_EXE) and not os.path.isfile(os.path.join(os.environ.get("PATH", "").split(":")[0], VINA_EXE)):
            raise FileNotFoundError(f'Vina executable not found: {VINA_EXE}')
        score, pose = self._run_vina(mode=mode, exhaustiveness=exhaustiveness)
        
        if not save_pose: 
            return score
        else: 
            return score, pose


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        ligand_rdmol = reconstruct_from_generated(data.clone())
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
        return [{'affinity': score, 'pose': pose}]


# if __name__ == '__main__':
#     lig_pdbqt = 'data/lig.pdbqt'
#     mol_file = 'data/1a4k_ligand.sdf'
#     a = PrepLig(mol_file, 'sdf')
#     # mol_file = 'CC(=C)C(=O)OCCN(C)C'
#     # a = PrepLig(mol_file, 'smi')
#     a.addH()
#     a.gen_conf()
#     a.get_pdbqt(lig_pdbqt)
#
#     prot_file = 'data/1a4k_protein_chainAB.pdb'
#     prot_dry = 'data/protein_dry.pdb'
#     prot_pqr = 'data/protein.pqr'
#     prot_pdbqt = 'data/protein.pdbqt'
#     b = PrepProt(prot_file)
#     b.del_water(prot_dry)
#     b.addH(prot_pqr)
#     b.get_pdbqt(prot_pdbqt)
#
#     dock = VinaDock(lig_pdbqt, prot_pdbqt)
#     dock.get_box()
#     dock.dock()
    
