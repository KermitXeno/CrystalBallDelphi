# CrystalBallDelphi
This is an active project that will eventually be an automated stock/crypto trading platform which will use a hierarchical model to time trade windows and assess risk. 
currently the planned tech stack will be:
- PHP (frontend/Backend)
- React(frontend)
- Python(backend (api(flask), models(Pytorch))

As of now we are using a temporally sensitive relationship graph and cascading market forecast output vectors from a regime predicton model (XGBRegressor) to a positional model for the purpose of composing trades. The model is evaluated on its ability to correctly time trade windows and make profit over short term trades within a 4 hour window, as of now the model is incapable of short selling due to regulatory restrictions on crypto options contracts.
