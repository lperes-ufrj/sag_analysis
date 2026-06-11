setup metacat
setup rucio
export RUCIO_ACCOUNT=justinreadonly
setup justin
justin time # this just tells justin that you exist and want to authenticate
justin get-token # this actually gets a token and associated proxy for access to rucio and the batch system