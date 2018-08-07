var express=require("express"); 
var cors=require("cors");
var bodyParser=require("body-parser");
var path=require("path");
var fs=require("fs");

//var popup = require('popups');

var mysql=require('mysql');
/*var connection = mysql.createConnection({
    //Gearhost mysql databse
  host     : 'den1.mysql6.gear.host',
  user     : 'winlabiot',
  password : 'winlabiot+123',
  database : 'winlabiot'
});*/

//connection.connect();

function insert(email){

    var connection = mysql.createConnection({
    //Gearhost mysql databse
    host     : 'den1.mysql6.gear.host',
    user     : 'winlabiot',
    password : 'winlabiot+123',
    database : 'winlabiot'
    });

    connection.connect();

    sql="INSERT IGNORE INTO coffee_mailing_list values ('"+email+"');"
        //connection.connect();
        connection.query(sql,function(err,rows,fields){
            if(err){
                throw err;
            }
        });
    connection.end();


}

//create express instance
var app=express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

//custom middleware to log message
app.use(function(req,res,next){
	
	var now = new Date(); 
	var datetime = (now.getMonth()+1) + "/"
                + now.getDate()  + "/" 
                + now.getFullYear() + "  "  
                + now.getHours() + ":"  
                + now.getMinutes() + ":" 
                + now.getSeconds();

    console.log(`${req.method} request for ${req.url}		${datetime}`);
    next();

});
 
//File Server on static files
app.use(express.static("./public"));

// add new email addr
app.post("/add", function(req, res){

    //console.log("here");
    console.log("Add email subscription: "+req.body.email);
    insert(req.body.email)
    

    //var email_addr=req.body.addr;

    //customer.storeToDB(connection);

    res.send({redirect: "/add.html"});

});








app.listen(8080);
console.log("Server running on port 8080");

module.exports=app;

