const API_URL = "http://127.0.0.1:8000";

const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");

const previewImage = document.getElementById("previewImage");
const previewContainer = document.getElementById("previewContainer");
const fileName = document.getElementById("fileName");

const predictionStatus = document.getElementById("predictionStatus");

let selectedFile = null;

let f1Chart = null;
let precisionRecallChart = null;


// ----------------------------------
// IMAGE SELECTION
// ----------------------------------

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) return;

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {

        alert("Please select a JPG, PNG or WEBP image.");

        imageInput.value = "";
        return;
    }

    selectedFile = file;

    previewImage.src = URL.createObjectURL(file);

    fileName.textContent = file.name;

    previewContainer.style.display = "block";

    predictBtn.disabled = false;

    predictionStatus.textContent = "";

    document.getElementById("emptyResult").hidden = false;
    document.getElementById("predictionResult").hidden = true;

});


// ----------------------------------
// IMAGE PREDICTION
// ----------------------------------

predictBtn.addEventListener("click", async function () {

    if (!selectedFile) return;

    const formData = new FormData();

    formData.append("file", selectedFile);

    predictBtn.disabled = true;

    predictBtn.textContent = "Analyzing...";

    predictionStatus.textContent = "Processing image using MobileNetV2...";

    try {

        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        displayPrediction(data);

        predictionStatus.textContent = "Analysis completed successfully.";

    }

    catch (error) {

        console.error(error);

        predictionStatus.textContent = error.message;

    }

    finally {

        predictBtn.disabled = false;

        predictBtn.textContent = "Analyze Image →";

    }

});


// ----------------------------------
// DISPLAY PREDICTION
// ----------------------------------

function displayPrediction(data) {

    document.getElementById("emptyResult").hidden = true;

    document.getElementById("predictionResult").hidden = false;

    document.getElementById("predictedClass").textContent =
        data.predicted_class;

    document.getElementById("confidence").textContent =
        `${data.confidence.toFixed(2)}%`;


    const probabilityList = document.getElementById("probabilityList");

    probabilityList.innerHTML = "";


    const sortedProbabilities = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);


    sortedProbabilities.forEach(([category, probability]) => {

        const item = document.createElement("div");

        item.className = "probability-item";

        const label = document.createElement("div");

        label.className = "probability-label";

        const categoryName = document.createElement("span");
        categoryName.textContent = category;

        const score = document.createElement("span");
        score.textContent = `${probability.toFixed(2)}%`;

        label.appendChild(categoryName);
        label.appendChild(score);


        const track = document.createElement("div");

        track.className = "progress-track";

        const fill = document.createElement("div");

        fill.className = "progress-fill";

        fill.style.width = `${probability}%`;

        track.appendChild(fill);

        item.appendChild(label);
        item.appendChild(track);

        probabilityList.appendChild(item);

    });

}


// ----------------------------------
// LOAD MODEL EVALUATION
// ----------------------------------

async function loadEvaluation() {

    try {

        const response = await fetch(`${API_URL}/evaluation`);

        if (!response.ok) {
            throw new Error("Could not load evaluation results");
        }

        const data = await response.json();

        displayMetrics(data);

        displayClassificationReport(data);

        createCharts(data);

    }

    catch (error) {

        console.error("Evaluation error:", error);

        document.getElementById("accuracy").textContent = "Error";

    }

}


// ----------------------------------
// DISPLAY METRICS
// ----------------------------------

function displayMetrics(data) {

    document.getElementById("accuracy").textContent =
        `${(data.accuracy * 100).toFixed(2)}%`;

    document.getElementById("precision").textContent =
        `${(data.macro_precision * 100).toFixed(2)}%`;

    document.getElementById("recall").textContent =
        `${(data.macro_recall * 100).toFixed(2)}%`;

    document.getElementById("f1").textContent =
        `${(data.macro_f1 * 100).toFixed(2)}%`;

    document.getElementById("testImages").textContent =
        data.test_images;

    document.getElementById("classCount").textContent =
        data.number_of_classes;

    document.getElementById("weightedF1").textContent =
        `${(data.weighted_f1 * 100).toFixed(2)}%`;

}


// ----------------------------------
// CLASSIFICATION REPORT TABLE
// ----------------------------------

function displayClassificationReport(data) {

    const report = data.classification_report;

    const reportBody = document.getElementById("reportBody");

    reportBody.innerHTML = "";

    data.class_names.forEach(category => {

        const result = report[category];

        const row = document.createElement("tr");

        const values = [
            category,
            `${(result.precision * 100).toFixed(2)}%`,
            `${(result.recall * 100).toFixed(2)}%`,
            `${(result["f1-score"] * 100).toFixed(2)}%`,
            result.support
        ];

        values.forEach(value => {

            const cell = document.createElement("td");

            cell.textContent = value;

            row.appendChild(cell);

        });

        reportBody.appendChild(row);

    });

}


// ----------------------------------
// CREATE CHARTS
// ----------------------------------

function createCharts(data) {

    const report = data.classification_report;

    const categories = data.class_names;

    const precision = categories.map(
        name => report[name].precision
    );

    const recall = categories.map(
        name => report[name].recall
    );

    const f1 = categories.map(
        name => report[name]["f1-score"]
    );


    // F1 SCORE CHART

    if (f1Chart) f1Chart.destroy();

    f1Chart = new Chart(
        document.getElementById("f1Chart"),
        {
            type: "bar",

            data: {
                labels: categories,

                datasets: [{
                    label: "F1-Score",

                    data: f1,

                    backgroundColor: "#36d399",

                    borderRadius: 5
                }]
            },

            options: {
                responsive: true,

                maintainAspectRatio: true,

                scales: {
                    y: {
                        min: 0,
                        max: 1,

                        ticks: {
                            color: "#8999af"
                        },

                        grid: {
                            color: "#26364d"
                        }
                    },

                    x: {
                        ticks: {
                            color: "#8999af"
                        },

                        grid: {
                            display: false
                        }
                    }
                },

                plugins: {
                    legend: {
                        labels: {
                            color: "#edf3fa"
                        }
                    }
                }
            }
        }
    );


    // PRECISION VS RECALL

    if (precisionRecallChart) {
        precisionRecallChart.destroy();
    }

    precisionRecallChart = new Chart(
        document.getElementById("precisionRecallChart"),
        {
            type: "bar",

            data: {
                labels: categories,

                datasets: [

                    {
                        label: "Precision",

                        data: precision,

                        backgroundColor: "#36d399",

                        borderRadius: 4
                    },

                    {
                        label: "Recall",

                        data: recall,

                        backgroundColor: "#6098e8",

                        borderRadius: 4
                    }

                ]
            },

            options: {

                responsive: true,

                scales: {

                    y: {
                        min: 0,
                        max: 1,

                        ticks: {
                            color: "#8999af"
                        },

                        grid: {
                            color: "#26364d"
                        }
                    },

                    x: {
                        ticks: {
                            color: "#8999af"
                        },

                        grid: {
                            display: false
                        }
                    }

                },

                plugins: {

                    legend: {
                        labels: {
                            color: "#edf3fa"
                        }
                    }

                }

            }

        }
    );

}


// ----------------------------------
// INITIALIZE DASHBOARD
// ----------------------------------

loadEvaluation();