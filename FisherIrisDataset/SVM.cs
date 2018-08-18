using System;
using System.Linq;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;

namespace FisherIrisDataset
{
    public class SVM
    {
        // 線形でクラス分けをする
        MulticlassSupportVectorMachine<Linear> msvm;

        // 茎の太さとか長さとかを入れる
        double[][] inputs { get; set; }
        // つまりクラス
        int[] outputs { get; set; }

        public SVM(double[][] inputs, int[] outputs)
        {
            this.inputs = inputs;
            this.outputs = outputs;
        }

        public void Learn(){
            var kernel = new Linear();
            var classes = outputs.GroupBy(x => x).Count();
            msvm = new MulticlassSupportVectorMachine<Linear>(0, kernel, classes);
            var teacher = new MulticlassSupportVectorLearning<Linear>(msvm)
            {
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // 並列計算の禁止
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            teacher.Learn(inputs, outputs);
        }

        // 判定をする
        public int Predict(double[] data){
            return msvm.Decide(data);
        }
    }
}
