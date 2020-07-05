var express = require("express");
const path = require("path");

var app = express();

var session = require("express-session"); // For sessions
var bodyParser = require("body-parser"); //
var multer = require("multer");

const exec = require("child_process").exec; //child process

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/images");
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + file.originalname);
  },
});

var upload = multer({ storage: storage });

app.set("trust proxy", 1);
app.use(
  session({
    secret: "7qbqgWUS;#^TRqU(",
    resave: false,
    saveUninitialized: true,
    cookie: {
      expires: 60000,
    },
    user_id: "",
    admin: false,
  })
);

app.use(express.static(path.join(__dirname, "css")));
app.use("/css", express.static(__dirname + "/css"));
app.use("/js", express.static(__dirname + "/js"));
app.use(express.static(path.join(__dirname, "js")));
app.use("/html", express.static(__dirname + "/html/public"));
app.use("/uploads/images", express.static(__dirname + "/uploads/images"));
app.use(express.static(path.join(__dirname, "/uploads/images")));

app.use(bodyParser.json());

app.get("/", function (req, res) {
  res.sendFile(path.join(__dirname, "/html/index.html"));
});

app.post("/upload", upload.single("photo"), (req, res) => {
  if (req.file) {
    res.json(req.file);
  } else throw "error";
});

app.post("/processImage", (req, res) => {
    /*
    * Method for Processing Images that come in right here! ;)
    * */
    let imagename = req.body.path
    imagename = /[^/]*$/.exec(imagename)[0];
    imagename = unescape(imagename)
    if (req.body) {
        exec(
            "./venv/bin/python3.8 analyze.py " + "\"" + imagename + "\"",
            (err, consoleOutput, consoleError) => {
                console.log(consoleOutput);
                console.log(consoleError);
                console.log(err);

            }
        );
  } else {
    res.redirect("/");
  }
});

app.post("/api/processImage", upload.single("photo"), (req, res) => {
    if (req.file.path) {
        exec(
            "python .... parameter!",
            (err, consoleOutput, consoleError) => {

            }
        );
    } else {
        res.redirect("/");
    }
});

app.listen(3000, function () {
  console.log("listening on *:3000");
});

/*Build input Pipeline! - */
