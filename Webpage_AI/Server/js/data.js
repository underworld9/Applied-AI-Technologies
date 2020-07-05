var app = angular.module("DataManagerApp", []);

function uploadImage(e) {
  e.preventDefault();
  let xhr = new XMLHttpRequest();
  xhr.open("post", "/upload", false);
  var formData = new FormData();
  let file = document.getElementById("fileid").files[0];
  formData.append("photo", file);
  xhr.send(formData);
  document.getElementById("bild").src = JSON.parse(xhr.response).path;
  document.getElementById("analyze").style.display="inline";
  return false;
}

function analyzeImage() {
  let xhr2 = new XMLHttpRequest();
  xhr2.open("post", "/processImage", false);
  xhr2.setRequestHeader('Content-Type', 'application/json')
  var path = document.getElementById("bild").src
  xhr2.send(JSON.stringify({path: path})); // path to be sent
  if(xhr2.status == 200){
    // bullshitverarbeituzng
    console.log(xhr.response);
  }
}