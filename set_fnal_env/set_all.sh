source set_protodune_vd.sh
source dune_token.sh
source dune_data.sh
echo $USER
export INPUT_TAR_DIR_LOCAL=`justin-cvmfs-upload ../pdvd_decoder_gallery_newfirmware.tar`
#export INPUT_TAR_DIR_LOCAL=/cvmfs/fifeuser2.opensciencegrid.org/sw/dune/6b669427c337376a060d5e4a47a17f3efd81a009
export FCL_FILE="$INPUT_TAR_DIR_LOCAL/standard_reco_stage1_protodunevd_keepup.fcl"
export FNALURL="https://fndcadoor.fnal.gov:2880/"
export USERF="dune/scratch/users/$USER/sag_analysis"
export SAG_SC="/pnfs/dune/scratch/users/$USER/sag_analysis"
export SAG_PERS="/pnfs/dune/persistent/users/$USER/sag_analysis"
export SAG_DIR="/exp/dune/app/users/$USER/sag_analysis"
#source /nashome/d/dbrailsf/setupVNCNew.sh
#export DISPLAY=:56159