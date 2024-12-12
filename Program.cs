namespace MachineLearning
{
    using Microsoft.ML;
    using Microsoft.ML.Data;

    // Definizione della struttura dei dati di input
    public class DatiCasa
    {
        [LoadColumn(0)]
        public float Dimensione { get; set; }

        [LoadColumn(1)]
        public float NumeroStanze { get; set; }

        [LoadColumn(2)]
        public float Prezzo { get; set; }
    }

    // Definizione della struttura di output per la predizione
    public class PredizioneCasa
    {
        [ColumnName("Score")]
        public float Prezzo { get; set; }
    }

    internal class Program
    {
        static void Main(string[] args)
        {
            // Creazione del contesto ML con seed fisso
            var contesto = new MLContext(seed: 0);

            var path1 = System.AppDomain.CurrentDomain.BaseDirectory+"""dataset\case.csv""";
            // Caricamento dei dati di esempio
            var dati = contesto.Data.LoadFromTextFile<DatiCasa>(
                path: path1,
                hasHeader: true,
                separatorChar: ',');

            // Creazione della pipeline di training
            var pipeline = contesto.Transforms.Concatenate("Features",
                    new[] { "Dimensione", "NumeroStanze" })
                .Append(contesto.Regression.Trainers.Sdca(
                    labelColumnName: "Prezzo",
                    maximumNumberOfIterations: 100));

            // Addestramento del modello
            var modello = pipeline.Fit(dati);

            // Creazione del motore di predizione
            var motorePredizione = contesto.Model.CreatePredictionEngine<DatiCasa, PredizioneCasa>(modello);

            // Effettua una predizione
            var casaEsempio = new DatiCasa()
            {
                Dimensione = 1200f,
                NumeroStanze = 3f
            };

            var predizione = motorePredizione.Predict(casaEsempio);
            Console.WriteLine($"Prezzo previsto: €{predizione.Prezzo:N0}");

            // Salvataggio del modello
            ////contesto.Model.Save(modello, dati.Schema, "PrezzoCasa.zip");
        }
    }
}
