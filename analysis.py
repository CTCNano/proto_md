import MDAnalysis as md
import numpy as np
import matplotlib.pylab as plt
import util

def loop(U, SS):

    U.trajectory.next()
    var_cg_t = [ss.frame() for ss in SS]

    return var_cg_t, U

def plotTimeIntegral(pdb, traj, var, ofname, subsystem_factory, plot, sel, subsystem_args, nCGplot, nframes):
    print 'Reading input ...'
    print sel
    U = md.Universe(pdb,traj)

    print 'Number of atoms is {}'.format(U.atoms.numberOfAtoms())
    print 'Size of atomic box is: \n {}'.format(U.atoms.bbox())
    print 'Read a total of {} frames'.format(nframes)

    trials = nframes
    factory = util.get_class(subsystem_factory)

    SS = factory(U, sel, **subsystem_args)
    
    [ss.universe_changed(U) for ss in SS]
    [ss.equilibriated() for ss in SS]

    if nCGplot is None:
	nCGplot = SS[0].NumNodes()

    global_nframes = 0
    U.trajectory.rewind()

    print 'Computing the time integral for a total of {} md runs'.format(trials)
    fp = [open('{}{}'.format(ofname,i),'w') for i in range(len(SS))]

    var_cg = [ss.frame() for ss in SS]
    
    for i in xrange(trials):
        var_cg_t, U = loop(U, SS)

	for n_ss in xrange(len(var_cg_t)):
        	np.savetxt(fp[n_ss], (var_cg_t[n_ss][0] - var_cg[n_ss][0]) / (i+1.0))
    
    [fpp.close() for fpp in fp]
    
    if plot:
	for n_ss, ss in enumerate(SS):
        	integral = np.loadtxt('{}{}'.format(ofname,n_ss))
		delta = np.arange(trials)
        
        	for order in xrange(N_CG_plot):
            		plt.plot(delta,integral[order:integral.shape[0]:N_CG_plot])
            		plt.show()
        
    print 'Done! Output successfully written to {}'.format(ofname)
