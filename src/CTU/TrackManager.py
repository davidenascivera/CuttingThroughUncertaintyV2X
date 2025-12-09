import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from SBE_module.SME_module import SMEModule

import logging
logger = logging.getLogger(__name__)


class TrackManager:
    def __init__(self, dA1:int, dR1:int, measurements: dict[int, dict[str, any]], sme_dict: dict[int, any],renderer = None, verbose:bool =False, id_counter:int  =0, color_dict:dict[int,str]={},
                 err_heading:float = 0, err_vel:float = 0) -> None:
        '''
        Initialize the TrackManager with measurements and SME dictionary.
        '''
        self.measurements = measurements
        self.SME_dict = sme_dict
        self.verbose = verbose
        self.renderer = renderer
        self.id_counter = id_counter
        self.color_dict = color_dict or {1000:'green', 2000:'yellow', 3000:'blue'}
        self.mature_track_keys = []
        self.dA1 = dA1
        self.dR1 = dR1
        self.err_heading = err_heading
        self.err_vel = err_vel

                   
    def process_measurements(self, dt:float):
        """Process all current measurements and update tracks"""
        # Update current_meas for this time step
        current_meas = {
            key: value
            for key, value in self.measurements.items()
            if value['dt'] == dt
        }
        
        # print(f"Processing measurements at dt: {dt} with {len(current_meas)} current measurements")
        # for k,v in current_meas.items():
        #     logger.info(f"Measure with id {k} - tracking ped {v['gt_id']} - sensor {v['sensor_id']} - track {v['track_id']}")
        # Associate measurements to mature tracks
        self._associate_with_mature_tracks(current_meas)
        
        # Associate remaining measurements with tentative tracks
        self._associate_with_tentative_tracks(current_meas)
        
        # Clean up old measurements
        self._clean_old_measurements()
        
        # Create or update SME for each measurement
        self._create_or_update_sme(dt)
        
        # Delete tracks marked for deletion
        self._delete_marked_tracks()
        
        return self.SME_dict, self.measurements, self.id_counter                
                 
    def _associate_with_mature_tracks(self, current_meas):
        """
        Associate current measurements with mature tracks.
        """
        # Get the list of mature tracks
        self.mature_track_keys = [k for k, v in self.SME_dict.items() if v.mature_track]
        
        # Building the matrix cost using only mature tracks
        ass_matrix = np.zeros((len(current_meas), len(self.mature_track_keys)))
        for row, meas in enumerate(current_meas):
            for col, track_id in enumerate(self.mature_track_keys):  # Use mature_track_keys instead of all SME_dict
                center_SME = self.SME_dict[track_id].centroid
                center_meas = current_meas[meas]['pos'] 
                
                distance = np.linalg.norm(np.array(center_meas)-np.array(center_SME))
                
                if distance > 1 or current_meas[meas]['sensor_id'] != self.SME_dict[track_id].sensor_id:
                    ass_matrix[row, col] = 1e6
                else:    
                    ass_matrix[row, col] = distance
                
 
                
        # hungarian alg with mature tracks:
        row_ind, col_ind = linear_sum_assignment(ass_matrix)
        meas_keys = list(current_meas.keys())
        
        if self.verbose:
            df = pd.DataFrame(ass_matrix, index=meas_keys, columns=self.mature_track_keys)
            logger.info("Association Matrix with IDs (mature tracks only): \n" + df.round(3).to_string())
        
        # Assign measurements to tracks
        for i, j in zip(row_ind, col_ind):
            mature_track_id = self.mature_track_keys[j]  # Convert index to actual track ID
            if self.verbose:
                logger.info(f"The pair is: {meas_keys[i]} -> {mature_track_id} | Cost: {ass_matrix[i, j]:.4f}")
            if ass_matrix[i, j] != 0 and ass_matrix[i, j]<1e5 : 
                self.measurements[meas_keys[i]]['track_id'] = mature_track_id
                
    def _associate_with_tentative_tracks(self, current_meas):
        tentative_tracks_keys = [k for k, v in self.SME_dict.items() if not v.mature_track]
        # Step 1: Get the unassigned measurements after mature track association
        unassigned_meas = {
            key: value
            for key, value in current_meas.items()
            if value['track_id'] is None  # Only include measurements that haven't been assigned
        }
        if self.verbose:
            for uns_meas in list(unassigned_meas.keys()):
                logger.info(f"Unassigned meas after mature track association: {uns_meas} gen by {self.measurements[uns_meas]['gt_id']} and sensor {self.measurements[uns_meas]['sensor_id']}" )

        # Step 2: Process tentative tracks association only if we have both unassigned measurements and tentative tracks
        if len(unassigned_meas) > 0 and len(tentative_tracks_keys) > 0:
            if self.verbose:
                logger.info(f"Starting tentative track association with {len(tentative_tracks_keys)} tentative tracks")
            
            # Build a new cost matrix for tentative tracks
            tent_ass_matrix = np.zeros((len(unassigned_meas), len(tentative_tracks_keys)))
            unassigned_meas_keys = list(unassigned_meas.keys())
            
            # Build the matrix with tentative tracks
            for row, meas_key in enumerate(unassigned_meas_keys):
                for col, track_id in enumerate(tentative_tracks_keys):
                    center_SME = self.SME_dict[track_id].centroid
                    center_meas = unassigned_meas[meas_key]['pos']
                    
                    distance = np.linalg.norm(np.array(center_meas)-np.array(center_SME))
                    if distance > 1.5 or current_meas[meas_key]['sensor_id'] != self.SME_dict[track_id].sensor_id:
                        tent_ass_matrix[row, col] = 1e6
                    else:    
                        tent_ass_matrix[row, col] = distance
                        # logger.debug(f"The distance is {distance:.3f} between meas {current_meas[meas_key]['meas_id']} and SME {self.SME_dict[track_id].track_id}. Pos meas: {center_meas}, pos centroid: {center_SME}")

            
            # # Plotting the matrix if need
            # df = pd.DataFrame(tent_ass_matrix)
            # logger.critical("TENTATIVE Matrix with IDs (mature tracks only): \n" + df.round(3).to_string())         
            
            
            # Apply Hungarian algorithm for tentative tracks
            tent_row_ind, tent_col_ind = linear_sum_assignment(tent_ass_matrix)
            
            if self.verbose:
                # Display tentative track association results
                tent_df = pd.DataFrame(tent_ass_matrix, index=unassigned_meas_keys, columns=tentative_tracks_keys)
                logger.info("Association Matrix with IDs (tentative tracks only):\n" + tent_df.round(3).to_string())
            
            # Process the assignments for tentative tracks
            for i, j in zip(tent_row_ind, tent_col_ind):
                tent_track_id = tentative_tracks_keys[j]
                meas_key = unassigned_meas_keys[i]
                
                
                if tent_ass_matrix[i, j] != 0 and tent_ass_matrix[i, j] < 1e5: # Only process if there's an actual association
                    self.measurements[meas_key]['track_id'] = tent_track_id
                    # self.SME_dict[tent_track_id].hits += 1
                    if self.verbose:
                        logger.info(f"Tentative pair: {meas_key} -> {tent_track_id} | Cost: {tent_ass_matrix[i, j]:.4f}")

    def _clean_old_measurements(self):
        '''
        We are generating the measurement with counter, so each one will be one after the other. we want to just keep the last measurement we used to 
        update a track, disregarding the others. To do so we check which is the most recent measurement track that has been assigned to the current tracks
        and remove all the others measurements. 
        '''
        all_track_keys = [k for k in self.SME_dict.keys()]
        to_keep = set()
        
        for track_id in all_track_keys:
        # Filtra tutte le entry in measurements con quel track_id
            matching = [(key, m) for key, m in self.measurements.items() if m['track_id'] == track_id]

            if not matching:
                continue  # Se non ci sono misure con quel track_id, salta

            # Trova la misura col dt più alto
            best = max(matching, key=lambda item: item[1]['dt'])  # item è (key, meas)
            to_keep.add(best[0])  # Salva solo la chiave (key)
    
        for key in list(self.measurements.keys()):
            if key not in to_keep and self.measurements[key]['track_id'] is not None:
                del self.measurements[key]
           
    def _create_or_update_sme(self, dt:float):
        """
        SME
        SME_dict: Dictionary containing SME (Set Membership Estimation) modules.
        Keys are track IDs (int), and values are SMEModule instances representing 
        the current reachable set estimation for each tracked object.
        
        SME_dict: dict[int, SMEModule]c

        """
        
        # current_meas = {k: v for k, v in self.measurements.items() if np.isclose(v['dt'], dt)}
        # for k,meas in current_meas.items():
        #     logger.info(f"Measure with id {k} - tracking ped {meas['gt_id']} - sensor {meas['sensor_id']} - track {meas['track_id']}")
        #     if meas['track_id'] is not None:
        #         logger.info(f"track {meas['track_id']} is from {self.SME_dict[meas['track_id']].id} and come from {self.SME_dict[meas['track_id']].sensor_id}")
        
        for meas_id, meas in self.measurements.items():
        # Create the filter 
            if meas['track_id'] is None and np.isclose(meas['dt'], dt): # se non è mai stato assegnato e se il tempo in cui è stato creato è lo stesso
                self.id_counter +=1
                meas['track_id'] = self.id_counter
                gt_ped = meas['gt_id']
                track_id = self.id_counter
                if self.verbose:
                    logger.info(f"Creating new SME for track_id {track_id} (dt: {dt}) with {meas_id} for ped {meas['gt_id']%10} with sensor {meas['sensor_id']} meas_id {meas_id}")
                self.SME_dict[track_id] = SMEModule(Xc = self.measurements[meas_id]['meas_set'], max_vel=1.8, acc=0.5, dt=0.1, ax=self.renderer.ax, delta_a=self.dA1, delta_r=self.dR1,
                                                plot_set=True, id = gt_ped, sens_id=meas['sensor_id'], colors=self.color_dict, plot_reach=True, plot_id=False,verbose = self.verbose, track_id = track_id,
                                                err_heading = self.err_heading, err_vel = self.err_vel) # This second row is for how well we want to overapproximate the circles
                self.SME_dict[track_id].predict(a_meas=np.rad2deg(self.measurements[meas_id]['head']), v_meas=self.measurements[meas_id]['vel'],dt_update=dt) # Predict the reachable set of the pedestrian
                
            
            # Update the filter
            ### NB: we have the reachable set from the previous time step and we are updating it with the new measurement.
            if meas['track_id'] is not None: # Else update the existing filter
                track_id = meas['track_id']
                
                if np.isclose(self.measurements[meas_id]['dt'], dt): # Check if we have a measurement to update the filter otherwis just run prediction
                    meas_set_update=self.measurements[meas_id]['meas_set']
                else:
                    meas_set_update = None
                

                # Prima questo era presente ma non ha senso. non so se la misura viene dal pedone. gt_id serve solo per debuggare
                
                if meas['gt_id'] != self.SME_dict[track_id].id:
                    continue
                
                self.SME_dict[track_id].intersect(meas_set=meas_set_update, lw_int=1, lw_prev_reach=0,dt_update=dt)
                measured_angle = np.rad2deg(self.measurements[meas_id]['head']) + np.random.uniform(-self.err_heading/2, self.err_heading/2) # Add some noise to the angle
                noise = np.random.uniform(-self.err_vel / 2, self. err_vel/ 2)
                measured_vel = max(self.measurements[meas_id]['vel'] + noise, 0)
                self.SME_dict[track_id].predict(a_meas= measured_angle, v_meas=measured_vel, lw_pred=1,dt_update=dt) # Predict the reachable set of the pedestrian
                
                # renderer.log_to_csv(filename, [dt, Polygon(SME_dict[1003].Xc).area, Polygon(SME_dict[id_meas].polygon).area]) 
                # pos_gt = [v['pos'] for v in vru_objects if v['id']]
                
    def _delete_marked_tracks(self):
        """
        Delete tracks that are marked for deletion.
        """
        keys_to_delete = []
        for key, value in self.SME_dict.items():
            if value.ready_to_delete:
                if self.verbose:
                    logger.info(f"Deleting SME with ID {key}")
                try:
                    measuremnt_corr = next(k for k,v in self.measurements.items() if v['track_id'] == key)
                    self.measurements[measuremnt_corr]['track_id'] = None
                    keys_to_delete.append(key)
                except StopIteration:
                    # Handle case where no measurement has this track_id
                    logger.warning(f"No measurement found with track_id {key}")
                    keys_to_delete.append(key)

        # Now delete the tracked items after the loop is done
        for key in keys_to_delete:
            if key in self.SME_dict:  # Check if key still exists
                del self.SME_dict[key]