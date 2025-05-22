import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment

import logging
logger = logging.getLogger(__name__)

class DataAssociationManager:
    '''
    ricordati che devi controllare anche quando hai avuto l'ultima misura.
    '''
    def __init__(self, id_sens_1:int, id_sens_2:int, measurements: dict[int, dict[str, any]], sme_dict: dict[int, any], 
                 blacklist: set[frozenset], mature_tracks_id: list, dt:float) -> None:
        
        self.sme_dict = sme_dict
        self.valid_assignments = []
        self.measurements = measurements
        
        mature_tracks = {k:v for k,v in sme_dict.items() if v.mature_track}
        self.track_sens_1 = {k:v for k,v in mature_tracks.items() if v.sensor_id == id_sens_1}
        self.track_sens_2 = {k:v for k,v in mature_tracks.items() if v.sensor_id == id_sens_2}

        self.n_1 = len(self.track_sens_1)
        self.n_2 = len(self.track_sens_2)        
        
        # Import the blacklist and compute the intial matrix.
        self.blacklist = blacklist
        self.cost_matrix =  np.zeros((self.n_1,self.n_2)) # Create the cost matrix. Now it is non padded

    
    def compute_cost_matrix(self, verbose: bool = False, verbose_prepad:bool = False) -> np.ndarray:
        '''
            We do the data association creating the a matrix with on the row the sensor 1 and on the column the sensor 2.
                       2001       2002
                    ------------------------
            1001 |  1001x2001   1001x2002
            1002 |  1002x2001   1002x2002

            Then if the intersection is = 0 we set the cost to inf and we add the pair to the blacklist.
            The dummy cost is set to 0.0 and the cost matrix is padded to make it square. 
        
        '''
        id_row = [v.id + v.sensor_id for v in self.track_sens_1.values()]
        id_col = [v.id + v.sensor_id for v in self.track_sens_2.values()]
        
        for row, (key1, value1) in enumerate(self.track_sens_1.items()):
            for col, (key2, value2) in enumerate(self.track_sens_2.items()):
                # if frozenset([key1, key2]) in self.blacklist:
                #     self.cost_matrix[row, col] = 1e6
                # else:

                #     poly1 = Polygon(value1.Xc)
                #     poly2 = Polygon(value2.Xc)
                    
                    
                #     # Compute the area of the intersection polygon
                #     area = poly1.intersection(poly2).area
                #     if np.isclose(area, 0):
                #         self.cost_matrix[row, col] = 1e6
                #         self.blacklist.add(frozenset([key1, key2]))
                #     else:
                #         IOU = area / (poly1.area + poly2.area - area)
                #         self.cost_matrix[row, col] = -IOU
                
             
                poly1 = self.sme_dict[key1].centroid
                poly2 = self.sme_dict[key2].centroid
                
                distance = np.linalg.norm(np.array(poly1)-np.array(poly2))
                self.cost_matrix[row, col] = distance
                    
                
        #Step 3: Pad the cost matrix
        #padded_cost_matrix = self.cost_matrix
        padded_cost_matrix = self.cost_matrix_padding(self.cost_matrix, dummy_cost=0.0, verbose=verbose_prepad)
        if verbose:
            # prendo copie di riga/colonna
            rows = id_row.copy()
            cols = id_col.copy()

            # calcolo quanti dummy servono
            row_pad = padded_cost_matrix.shape[0] - len(rows)
            col_pad = padded_cost_matrix.shape[1] - len(cols)

            # genero etichette dummy univoche
            for i in range(row_pad):
                rows.append(f"dummy_row_{i}")
            for j in range(col_pad):
                cols.append(f"dummy_col_{j}")

            # uso padded, non self.cost_matrix “non–padded”
            cost_mat = pd.DataFrame(padded_cost_matrix, index=rows, columns=cols)
            logger.info(f"Cost matrix after padding: \n{cost_mat}")
        self.cost_matrix = padded_cost_matrix
        
        return self.blacklist
    
    def solve(self, verbose: bool = False) -> list[tuple[int, int]]:
        """
        Greedy data association: per ogni cella (i,j) del cost_matrix, ordina
        tutte le coppie per costo crescente e associa riga e colonna se non sono
        già state usate, fermandosi quando non ci sono più righe/colonne libere.
        """
        # assicuriamoci di aver già chiamato compute_cost_matrix()
        cost_mat = self.cost_matrix.copy()
        n_rows, n_cols = cost_mat.shape

        # liste di chiavi originali per mapparle indici->track_id
        track_sens_1_keys = list(self.track_sens_1.keys())
        track_sens_2_keys = list(self.track_sens_2.keys())

        # prepariamo tutti i tuple (riga, colonna, costo)
        flat = []
        for i in range(n_rows):
            for j in range(n_cols):
                flat.append((i, j, cost_mat[i, j]))
        # ordiniamo per costo crescente
        flat.sort(key=lambda x: x[2])

        assigned_rows = set()
        assigned_cols = set()
        valid = []

        for i, j, cost in flat:
            # consideriamo solo righe/colonne "vere" (non dummy) e cost < inf
            if i < self.n_1 and j < self.n_2 and cost < 1e6:
                if i not in assigned_rows and j not in assigned_cols:
                    id1 = track_sens_1_keys[i]
                    id2 = track_sens_2_keys[j]
                    valid.append((id1, id2))
                    assigned_rows.add(i)
                    assigned_cols.add(j)
                    if verbose:
                        logger.info(f"Greedy pair: {id1} ↔ {id2} | cost {cost:.3f}")
            # se tutte le righe o tutte le colonne sono state assegnate, usciamo
            if len(assigned_rows) >= self.n_1 or len(assigned_cols) >= self.n_2:
                break

        self.valid_assignments = valid
        return valid
    
     
    def validate(self) -> list[tuple[int, int]]:
        '''
        Function to validate the assignments. It checks if the association is correct.
        '''
        # Step 1: Check for errors:
        wrong_assignments = []
        for ass in self.valid_assignments:
            track1 = self.track_sens_1.get(ass[0])
            track2 = self.track_sens_2.get(ass[1])
            
            if track1 is None or track2 is None:
                logger.warning(f"Track not found for assignment {ass}")
                continue
                
            # Assuming each track has a 'gt_id' attribute
            gt_id_1 = track1.id if hasattr(track1, 'id') else None
            gt_id_2 = track2.id if hasattr(track2, 'id') else None
            
            if gt_id_1 is not None and gt_id_2 is not None:
                if gt_id_1 % 10 != gt_id_2 % 10:
                    wrong_assignments.append(ass)
                    logger.error(f"WRONG ASSOCIATION {gt_id_1} with {gt_id_2}")
            else:
                logger.warning(f"Missing gt_id for assignment {ass}")
        
        return wrong_assignments

    
        
        
    @staticmethod
    def cost_matrix_padding(cost_mat: np.ndarray, dummy_cost: float = 0.0, verbose: bool = False) -> np.ndarray:
        '''
        Ensures the cost matrix is square by padding with dummy rows/columns.
        Also handles all-inf rows/columns to avoid assignment issues.
    
        Parameters:
            cost_mat (np.ndarray): The input cost matrix.
            dummy_cost (float): Value for dummy entries (default 0.0).
            verbose (bool): If True, prints debug info.
    
        Returns:
            np.ndarray: The padded cost matrix.
        '''
        n_rows, n_cols = cost_mat.shape

        # Step 1: Padding the cost matrix to make it square
        if n_rows > n_cols:
            diff = n_rows - n_cols
            if verbose:
                print(f"Padding {diff} dummy column(s)")
            dummy_cols = np.full((n_rows, diff), dummy_cost)
            cost_mat = np.hstack((cost_mat, dummy_cols))
        elif n_cols > n_rows:
            diff = n_cols - n_rows
            if verbose:
                print(f"Padding {diff} dummy row(s)")
            dummy_rows = np.full((diff, n_cols), dummy_cost)
            cost_mat = np.vstack((cost_mat, dummy_rows))

        # Step 2: Protezione contro righe interamente inf
        inf_rows = np.where(np.all(np.isinf(cost_mat), axis=1))[0]
        if inf_rows.size > 0:
            if verbose:
                print(f"Found {len(inf_rows)} all-inf row(s), padding extra dummy row/col")
            cost_mat = np.vstack((cost_mat, np.full((1, cost_mat.shape[1]), dummy_cost)))
            cost_mat = np.hstack((cost_mat, np.full((cost_mat.shape[0], 1), dummy_cost)))

        # Step 3: Protezione contro colonne interamente inf
        inf_cols = np.where(np.all(np.isinf(cost_mat), axis=0))[0]
        if inf_cols.size > 0:
            if verbose:
                print(f"Found {len(inf_cols)} all-inf column(s), padding extra dummy row/col")
            cost_mat = np.vstack((cost_mat, np.full((1, cost_mat.shape[1]), dummy_cost)))
            cost_mat = np.hstack((cost_mat, np.full((cost_mat.shape[0], 1), dummy_cost)))

        return cost_mat

def main():
    # Step1: Create the cost matrix
    pass
if __name__ == "__main__":
    main()
      

      
      
      
      
      
      
