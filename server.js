import express from "express";
import cors from "cors";
import { createServer } from "http";
import { Server } from "socket.io";

const app = express();

app.use(cors());

const httpServer = createServer(app);
const io = new Server(httpServer, { 
  cors: {
    origin: false
  }
 });

app.get('/', (req, res) => {
    res.send('Hello, World!');
})

io.on("connection", (socket) => {
  console.log("Socket connected: " + socket.id);

  socket.on("disconnect", () => {
    console.log(socket.id + " disconnected.");
  })

  socket.onAny((eventName, data) => {
    console.log("Caught: " + eventName);
    console.log(data);
    socket.emit(eventName, data);
  });
});

httpServer.listen(3000, () => {
    console.log("OK");
});