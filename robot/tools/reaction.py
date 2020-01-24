import time

from robot.utils import rxn_utils

class Reaction(object):

    def __init__(self, experiment):

        self.exp = experiment
        self.coords = self.exp.robot.plat.coords        # get coordinates

        self.XY = self.exp.robot.XY                     # define access to XY motors
        self.Z = self.exp.robot.Z                       # define access to Z motors

        self.cont = self.exp.robot.cont                 # define access to pumps
        self.reaction = rxn_utils.order_reagents(self.exp.reagent_data) # define the reaction procedure


    def run(self, rxn_number):

        self.pump_reagents()                # get reagents into syringe
        print('xy to {}'.format(self.coords[rxn_number]))
        self.XY.move_to(self.coords[rxn_number])    #   move arm
        self.Z.move_to(self.exp.robot.plat.z_depth) #   move arm
        self.deliver_reagents()             # deliver reagents from syringe
        self.Z.home()



    def pump_reagents(self):
        # component[0] is the name of the pump, component[1] is a dict of the parameters to employ
        for component in self.reaction:
            if component[1]['dispense']:
                print('pumping {}'.format((component[0], component[1]['volume'])))
                self.cont.pumps[component[0]].pump(component[1]['volume'], 'I', wait=False)
        self.cont.wait_until_all_pumps_idle()

    def deliver_reagents(self):
        # ensure all pumps are ready to dispense
        self.cont.wait_until_all_pumps_idle()
        # start a timer to allow dispensing at correct time
        start_time = time.time()
        for component in self.reaction:
            if component[1]['dispense']:
                elapsed_time = time.time() - start_time
                if elapsed_time < component[1]['time']:
                    time.sleep(float(component[1]['time']) - elapsed_time)
                print('delivering {}'.format((component[0], component[1]['volume'])))
                self.cont.pumps[component[0]].deliver(component[1]['volume'], 'O', wait=False)
        # ensure all pumps are ready to dispense
        self.cont.wait_until_all_pumps_idle()
