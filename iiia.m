%% III.a Verify derivative of Sigmoidal fxn

syms x
f = 1/(1 + exp(-x));
diff(f)