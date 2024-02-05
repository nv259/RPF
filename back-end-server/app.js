const express = require('express');
const multer = require('multer');
const path = require('path');
const request = require("request");
const cors = require("cors");
const app = express();
const port = 4000;

app.use(express.static("uploads"));
app.use(cors());

// Set up multer for handling file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './uploads'); // Specify the folder where you want to save the uploaded files
  },
  filename: (req, file, cb) => {
    cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage });

// Serve the HTML form
// (app.get('/', (req, res) => {
//   res.sendFile(path.join(__dirname, 'index.html'));
// })

// Handle file upload
app.post('/submit', upload.single("image"), async (req, res) => {
  data = {
    "img_path": req.protocol + "://" + req.get("host") + "/" + req.file.filename,
    "attrs": req.body.attributes
  }
  await request.post(
    uri="http://localhost:8000/submit",
    { json: data },
    function (error, response, body) {
      console.log(body);
    }
  )
  // res.send('Image uploaded successfully!');
  res.json({"message": "image uploaded successfully!"});
});

axios.post("http://localhost:8000/submit", data).then((response) => 
{
  console.log(response);
  res.send(response);
}
);



app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
