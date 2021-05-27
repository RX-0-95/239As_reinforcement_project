## Run tensorboradX to get plot use command 

            tensorboard --logdir=runs
    
## Run code in bash 

## Pong 
### Double Q learning 
            python3 dqn_pong.py --cuda --double --reward 21

### Deep Q learning 

            python3 dqn_pong.py --cuda --double False --reward 19

## Result 

* Pong (1.2*10^6 iter)
    - Double Q: 19.64
    - Q: 

    
## Aline 
### Double Q learning 
            python3 dqn_aline.py --cuda --double --reward 5000

### Deep Q learning 

            python3 dqn_aline.py --cuda --double False --reward 5000