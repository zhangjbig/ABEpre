# ABEpre
## 🚀 Welcome to ABEpre: Accurate, Robust, and Efficient B-cell Epitope Prediction Pipeline!

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

#### Instructions:

1\.	It is recommended to use Conda to replicate our environment.

        \*\*\*conda env create -n ABEpre -f conda.yml\*\*\*

2\.    Run with \*\*\* python ABEpre.py -type protein -infile input.txt -outfile ouput.txt\*\*\*

 	-type: Choose protein or peptide.

        -infile: Inputfile.

                    The standard format is that the first line contains a sequence identifier

 		    (which can include a name, description, etc.) beginning with ">".

 		    The second line contains the corresponding amino acid sequence

                    (which can be multiple lines, but will be automatically concatenated).

                    If the input file does not contain any label lines beginning with ">",

                    each line will be automatically treated as an anonymous sequence and assigned a default name

                   (such as Seq\_1, Seq\_2, etc.).

        -outfile: Outputfile.

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

#### If you want to reproduce the training process of our model:

1\.	Enter the Albert folder and run tokenizer.py to generate a word tokenizer.

         \*\*\*python tokenizer.py\*\*\*



2\.	Run Albert.py to generate a pre-trained Transformer model.

         \*\*\*python albert.py\*\*\*

3\.	Return to the original folder and run train.py to start  training process.

        \*\*\*python train.py\*\*\*

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

#### Other scripts provided:

1\.	albertdatapre.py  To generate input data for tokenizer.py and albert.py.

2\.	tuner\_para.py For Hyperparameter Search.

