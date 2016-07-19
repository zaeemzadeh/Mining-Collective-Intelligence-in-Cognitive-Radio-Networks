using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;

namespace MicrosoftResearch.Infer.Tutorials
{
    public class MiningCRN
    {
        public void Run()
        {
            int N = 8;  // number of devices
            int M = 6 ;  // number of channel
            int L = N; // group truncation 
            int T = 5; // number of time slots

            // defining range for each random variable
            Range n = new Range(N).Named("n");
            Range m = new Range(M).Named("m");
            Range l = new Range(L).Named("l");
            
            // Defining Latent Variables
            var R = Variable.Array<double>(n).Named("Device Reliability");
            var U = Variable.Array(Variable.Array<bool>(m), n).Named("Channel Specific Reliability");
            var G = Variable.Array<int>(n).Named("Cluster Assignment");
            var C = Variable.Array(Variable.Array<double>(m), l).Named("Channel Availibility for Cluster");
            var Y = Variable.Array(Variable.Array<bool>(m), n).Named("Observations");

            // Specifying priors for the latent variables
            var ReliabilityPrior = Variable.Array<Beta>(n).Named("Reliability Prior");
            var ClusterPrior = Variable.Array<Discrete>(n).Named("Cluster Prior");
            var ChannelPrior = Variable.Array(Variable.Array<Beta>(m), l).Named("Channel Prior");

            // Define latent variables statistically
            R[n] = Variable<double>.Random(ReliabilityPrior[n]);
            U[n][ m] = Variable.Bernoulli(R[n]).ForEach(m);
            G[n] = Variable<int>.Random(ClusterPrior[n]);
            G.SetValueRange(l);
            //G[n] = Variable.DiscreteUniform(l).ForEach(n);
            C[l][ m] = Variable<double>.Random(ChannelPrior[l][m]);
   
            // Stick breaking construction
            double[] group_weights = new double[L]; // initial probabilities for groups assignment
            double kappa = N;       // concentration factor dor stick breaking
            double p = 1;
            for (int i = 0; i < L; i++)
            {
                double rhoi = Rand.Beta(1, kappa);
                group_weights[i] = p*rhoi;
                p *= 1 - rhoi;
            }
            // Initialise priors 
            Beta ReliabilityDist = new Beta(5, 1);  // TODO: Set reliasitic hyperparameters
            Beta ChannelDist = new Beta(1, 1);
            Discrete ClusterDist = new Discrete(group_weights);

            ReliabilityPrior.ObservedValue = Util.ArrayInit(N, u => ReliabilityDist);
            ClusterPrior.ObservedValue = Util.ArrayInit(N, u => ClusterDist);
            ChannelPrior.ObservedValue = Util.ArrayInit(L, i => Util.ArrayInit(M, t => ChannelDist)); ;


            // data variables
            bool[][] d_data = new bool[N][];
            bool[][] d_missing = new bool[N][];
            for (int i = 0; i < N; i++)
            {
                d_data[i] = new bool[M];
                d_missing[i] = new bool[M];
                bool[] group1 = new bool[] { true, true, false ,false,false, false};
                bool[] group2 = new bool[] {  false, false, true, true, false, false };
                bool[] group3 = new bool[] { false, false, false, false, true, true, };

                for (int j = 0; j < M; j++)
                {
                    d_data[i][j] = group3[j];
                    if (i % 8 == 0)
                        d_data[i][j] = group1[j];

                    if (i % 4 == 1)
                        d_data[i][ j] = group2[j];

                    d_missing[i][j] = false;  //(i%2 == 0) && (j%2 == 0);
                }
            }
            var v_missing = Variable.Observed(d_missing, n, m);
            //VariableArray2D<bool> v_data = Variable.Observed(d_data, n, m);

            
            // Model
            using (Variable.ForEach(n))
            {
                using (Variable.Switch(G[n]))
                {
                    using (Variable.ForEach(m))
                    {
                        using (Variable.IfNot(v_missing[n][m]))
                        {
                            using (Variable.If(U[n][ m]))
                            {
                                Y[n][ m] = Variable.Bernoulli(C[G[n]][m]);   // reliable device
                            }
                            using (Variable.IfNot(U[n][ m]))
                            {
                                Y[n][ m] = !Variable.Bernoulli(C[G[n]][m]);  // unreliable device
                            }
                        }
                    }
                }
            }
            Y.ObservedValue = d_data;

            
                
            for (int t = 0; t < T; t++)
            {

                InferenceEngine ie = new InferenceEngine();
                //ie.ShowFactorGraph = true;
                ie.Algorithm = new VariationalMessagePassing();
                var ObseReliabilityPosterior = ie.Infer<Bernoulli[][]>(U);
                var ClusterPosterior = ie.Infer<Discrete[]>(G);
                var ReliabilityPosterior = ie.Infer<Beta[]>(R);
                var AvailibiltyPosterior = ie.Infer<Beta[][]>(C);
                //ReliabilityPrior.ObservedValue = obs_dist;
                Console.WriteLine("Device Reliability:");
                foreach (var r in ReliabilityPosterior) Console.WriteLine(r);
                Console.WriteLine("Channel Availibility:");
                foreach (var a in AvailibiltyPosterior) { foreach (var aa in a) Console.WriteLine(aa);  Console.WriteLine("");}
                Console.WriteLine("Cluster Assignment:");
                foreach (var c in ClusterPosterior) Console.WriteLine(c);
                ReliabilityPrior.ObservedValue = ReliabilityPosterior;
                ChannelPrior.ObservedValue = AvailibiltyPosterior;
                ClusterPrior.ObservedValue = ClusterPosterior;
            }
        }

        public void GetInfo(ref int N, ref int M, ref int T)
        {
            
        }
        public void GetData(int N, int M, int T, ref double[,,] d_data) {
            //d_data = 
        }

        static void Main(string[] args)
        {
            MiningCRN a = new MiningCRN();
            a.Run();
            Console.ReadKey();
        }

    } 

}

