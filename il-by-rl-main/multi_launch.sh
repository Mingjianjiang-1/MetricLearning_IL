for ENV in 'halfcheetah'; do
  for AL in 'ilbc' 'il' 'gmmil' 'gmmilfo' 'gail' 'sqil' 'bc'; do
     for SEED in 1 2 3; do
      sbatch launch.sh $ENV $AL $SEED &
    done &
  done &
done


#for ENV in 'ant'; do
#  for METHOD in 'ilbc' 'il' 'gmmil' 'gail' ''; do
#    for SEED in 1 2 3 4 5; do
#      for ALPHA in 0.1 0.5 1 2 5; do
#        sbatch launch.sh $ENV $METHOD $SEED $ALPHA '' &
#      done &
#    done &
#  done &
#done


