#! /usr/bin/env python2
import subprocess, time, os, sys

LOCAL_DIR = '/Users/danil/My/School/Columbia/Research/ongoing_plasticity/'
REMOTE_SYNC_SERVER = 'dt2586@habaxfer.rcs.columbia.edu' #must have ssh keys set up
REMOTE_SERVER = 'dt2586@habanero.rcs.columbia.edu'
REMOTE_PATH = '/rigel/home/dt2586/ongoing_plasticity' #'/rigel/theory/users/dt2586/ongoing_plasticity' 
REMOTE_DIR = '{}:{}'.format(REMOTE_SYNC_SERVER, REMOTE_PATH)

def sbatch(pyscript, args='', output=None, cpus=2, ram=16, gpu=False, folder=None, short=False): 
    """This method gets executed on the cluster"""
    if folder is not None: 
        execFolder = os.path.join(REMOTE_PATH, folder)
        if not os.path.exists(execFolder):
            os.makedirs(execFolder)
        sys.path.append(REMOTE_PATH) #append this to path so it can find the ongoing_plasticity library from different folder
    else:
        execFolder = REMOTE_PATH         
    os.chdir(execFolder)  
           
    BATCHSCRIPT = 'batchscript.sh' #TODO: make unique temporary file instead
    
    if ram > 128:
        print('[{}] WARNING: desired RAM={} >128 GB. This uses the high-memory node. Setting CPUS=1"'.format(sys.platform, ram))
        cpus = 1
    
    #default output filename if not specified
    if output is None:
        output = 'log_{}_{}.out'.format(os.path.basename(pyscript), str.replace(args, ' ', '_'))
    
    maxFilenameLength = 255
    if len(output)>maxFilenameLength:
        output = output[:maxFilenameLength-4]+'.out' 
    
    #ensure output filename is unique 
    base, ext = os.path.splitext(output)
    n = 2
    while os.path.exists(output): #TODO: this doesn't work when batching multiple jobs w/ same output filename 
        output = '{}_({}){}'.format(base, n, ext)
        n+=1 
    output = output.replace('/', '-')
    
    gpu = '#SBATCH --gres=gpu:1 \n' if gpu else '' #GPU config 
    short = '#SBATCH --time 0-12:00 \n' if short else '' #limit job to 12 hours but get added to "short" queue (usually starts sooner)
    content = (
    '#!/bin/sh \n' 
    '#SBATCH --account=theory \n'
    '#SBATCH --cpus-per-task={cpus} \n' 
    '#SBATCH --mem-per-cpu={mem}gb \n'  
#    '#SBATCH --partition=test \n' #use test partition for no wait time during debugging
#    '#SBATCH --time 1:00:00 \n' 
    '{short}' 
    '{gpu}'
    '#SBATCH --job-name={out} \n'
    '#SBATCH --output={out} \n\n' #TODO: can I also write to stdout in parallel? (to see it in local terminal)
     #TODO: notify via text or email when done?

    'echo [Batch script] Parsed args: {script} {args} \n'
    'stdbuf -oL python {script} {args} \n' 
    ).format(cpus=cpus, mem=ram, short=short, gpu=gpu, out=output, script=os.path.join(REMOTE_PATH, pyscript), args=args)
    
    with open(BATCHSCRIPT, 'w') as f:
        f.write(content)
      
    cmd = 'sbatch {batchscript}'.format(batchscript=BATCHSCRIPT)
    print('[{}] Batching "{} {}"'.format(sys.platform, pyscript, args))
    subprocess.check_call(cmd, shell=True)
    os.remove(BATCHSCRIPT)
    
    
def push_to_haba():
    print('[{}] Pushing to {}...'.format(sys.platform, REMOTE_SYNC_SERVER))
    cmd = "rsync -vv {local}*.py ~/My/dt_utils.py {remote}".format(local=LOCAL_DIR, remote=REMOTE_DIR)
    subprocess.check_call(cmd, shell=True) 

   
def pull_from_haba(folder=time.strftime("%Y-%m-%d-%H-%M")):
    print('[{}] Retrieving from {}...'.format(sys.platform, REMOTE_SYNC_SERVER))

    destination = '{}/results/{}/'.format(LOCAL_DIR, folder)
    if not os.path.exists(destination):
        os.makedirs(destination)    
    cmd = 'rsync -vv --remove-source-files {}*.pkl {}'.format(REMOTE_DIR, destination)
    subprocess.check_call(cmd, shell=True) 
    
     
def is_done():
    cmd = "ssh -q {} 'squeue -u dt2586 | wc -l'".format(REMOTE_SERVER)
    ret = subprocess.check_output(cmd, shell=True)
    qlen = int(ret)-1 #subtract 1 to account for row header
    if qlen == 0:
        print('[{}] Queue is empty!'.format(sys.platform))
        return True
    return False
      

if __name__ == '__main__':    
    print('[{}] Syncing and running on {}...'.format(sys.platform, REMOTE_SERVER))
    push_to_haba()
    
    cmd = "ssh -q {} 'python {}/haba_loop.py'".format(REMOTE_SERVER, REMOTE_PATH)
    subprocess.check_call(cmd , shell=True )
       
#    raw_input('[{}] Press Enter to exit...'.format(sys.platform))
    
#    while not is_done():
#        time.sleep(300)
#    pull_from_haba()     
    
 
        
        
