// Fonction de classification du texte
let chartInstance = null;
function classifyText() {
  var inputText = document.getElementById("text-input").value;

  var xhr = new XMLHttpRequest();
  xhr.open("POST", "http://localhost:5000/classify", true);
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  // Ajout du gestionnaire d'erreur réseau
  xhr.onerror = function () {
    console.error("Erreur réseau.");
    alert("Une erreur réseau s'est produite.");
  };

  // Quand la réponse est reçue
  xhr.onload = function () {
    if (xhr.status == 200) {
      var response = JSON.parse(xhr.responseText);

      // Afficher la classe prédite
      var result = response.predicted_class;
      var resultSection = document.getElementById("resultSection");

      // Afficher les probabilités
      var probabilities = response.class_probabilities;

      const label = Object.keys(probabilities); // ['business', 'entertainment', 'politics', 'sports', 'technology']
      const values = Object.values(probabilities); // [10, 20, 30, 40, 50]

      // Create the bar chart using Chart.js
      const ctx = document.getElementById("histogramCanva").getContext("2d");
      if (chartInstance) {
        chartInstance.destroy();
      }
      const data = {
        labels: label,
        datasets: [
          {
            label: "Categories",
            data: values,
            backgroundColor: "blue",
          },
        ],
      };

      chartInstance = new Chart(ctx, {
        type: "bar",
        data: data,
        options: {
          scales: {
            y: {
              beginAtZero: true,
            },
          },
        },
      });
    } else {
      console.error("Erreur :", xhr.statusText);
    }
  };

  // Envoyer la requête avec le texte
  var data = JSON.stringify({ text: inputText });
  xhr.send(data);
}

// Fonction pour effacer le texte et les résultats
function clearText() {
  document.getElementById("text-input").value = "";
  chartInstance.destroy();
}

// Fonction pour télécharger le fichier CSV après classification
function classifyAndDownload() {
  var fileInput = document.getElementById("file");
  var file = fileInput.files[0];

  if (file) {
    var formData = new FormData();
    formData.append("file", file);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/classify-and-download", true);
    xhr.responseType = "blob";

    xhr.onload = function () {
      if (xhr.status == 200) {
        var blob = new Blob([xhr.response], { type: "text/csv" });
        var link = document.createElement("a");
        link.href = window.URL.createObjectURL(blob);
        link.download = "resultats_classement.csv";
        link.click();
        console.log("Téléchargement du fichier terminé avec succès.");
      } else {
        console.error("Erreur :", xhr.statusText);
      }
    };

    xhr.onerror = function () {
      console.error("Erreur réseau.");
      alert("Une erreur réseau s'est produite.");
    };

    xhr.send(formData);
  } else {
    alert("Veuillez choisir un fichier à téléverser.");
  }
}

// Fonction pour uploader un fichier et envoyer à backend
function uploadFile() {
  var fileInput = document.getElementById("file");
  var file = fileInput.files[0];

  if (file) {
    var allowedTypes = [
      "text/plain",
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ];
    var fileExtension = file.name.split(".").pop().toLowerCase();
    if (
      !allowedTypes.includes(file.type) &&
      !["txt", "pdf", "docx"].includes(fileExtension)
    ) {
      alert("Seuls les fichiers TXT, PDF et DOCX sont autorisés.");
      return;
    }

    var formData = new FormData();
    formData.append("file", file);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/upload", true);

    xhr.onload = function () {
      if (xhr.status == 200) {
        alert("Fichier envoyé avec succès !");
      } else {
        console.error("Erreur :", xhr.statusText);
        alert("Erreur lors de l'envoi du fichier.");
      }
    };

    xhr.onerror = function () {
      console.error("Erreur réseau.");
      alert("Une erreur réseau s'est produite.");
    };

    xhr.send(formData);
  } else {
    alert("Veuillez sélectionner un fichier à envoyer.");
  }
}
