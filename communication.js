
 //d=[]
 function off() {
  document.getElementById("overlay").style.display = "none";
}
function showPage() {
  document.getElementById("loader").style.display = "none";
  //document.getElementById("myDiv").style.display = "block";
}
 function get()
 {
 	//var search=document.getElementById('query').value;
 	//alert("hello world");
 	document.getElementById("overlay").style.display = "block";
 	var myVar;
 	//myVar = setTimeout(showPage, 3000);
    $.ajax({
            url: '/signUpUser',
            data: {
                 query : $("#query").val()
                 },
            headers:{
               'Cache-Control': 'max-age=0'
            },
            dataType: "json",
            type: 'POST'

        })
        .done(function(data){
          //msg=data.name;
          //var result=JSON.parse(data);
           d=data.name;
           canvas(d);
           document.getElementById("overlay").style.display = "none";

          //alert(d[0]);
          //prompt(typeof msg)
          //msg=$.parseJSON(data);
          //alert("kundan");
          //alert(msg);

               });




 	//canvas();
 }
function explore()
 {
    var f=recieve_feedback();
    //alert(f);
    //alert(f.length);
    //alert(f[0]);
    //alert(f[1]);
    //alert(typeof f[1]);
    var str=f[0].toString();
    var i;
    for(var i=1;i<f.length;i++)
    {
        str=str+" "+f[i].toString();
    }
    //alert(str);
    var searc=document.getElementById('query').value;
    document.getElementById("overlay").style.display = "block";
 	var myVar;
    $.ajax({
            url: '/signUpUser2',
            data: {
                 query : str,
                 search : searc
                 },
            headers:{
               'Cache-Control': 'max-age=0'
            },
            dataType: "json",
            type: 'POST'

        })
        .done(function(data){
          //msg=data.name;
          //var result=JSON.parse(data);
           d=data.name;
           canvas(d);
           document.getElementById("overlay").style.display = "none";

          //alert(d[0]);
          //prompt(typeof msg)
          //msg=$.parseJSON(data);
          //alert("kundan");
          //alert(msg);

               });
    //alert(f)
 	//canvas();
 }

feedback=[];
function recieve_feedback()
 {
    return feedback;
 }
 function writeMessage(text,layer,x_cord,y_cord,message) {
        text.x(x_cord-20);
        text.y(y_cord-20);
        text.text(message);
        layer.draw();
        //alert(message);
      }
 function canvas(params)
 {
//var msg2="kundan";
//alert(params[0]);
//prompt(msg2);
if(typeof params[0][0]=="string")
    {alert("keyword not present in any file");
    document.getElementById("overlay").style.display = "none";
    }
    var Parent = document.getElementById('listtable');
while(Parent.hasChildNodes())
{
   Parent.removeChild(Parent.firstChild);
}
var i;
var tablearea2 = document.getElementById('listtable2');
    table = document.createElement('table');


for (var i = 0; i < 5 ; i++) {
    var tr = document.createElement('tr');
    var td1=document.createElement('td');
    tr.appendChild(td1);
    td1.setAttribute('text-align','left');
    td1.setAttribute('vertical-align','top');
    td1.setAttribute('align','left');
    //tr.appendChild( document.createElement('td') );
    //var h = document.createElement("mark")
    //var arr=['#ECA43E','#EF5B17','#B5D57C','#0066ff','#990000'];
    //var j=Math.floor(Math.random()*4)+1;
    //h.style.backgroundColor=arr[j-1];
    //h.style.borderRadius = "5px";
    //j=Math.floor(Math.random()*4)+1;
   // h.style.color=arr[j-1];
    var td =document.createTextNode(params[i][1]);
    //h.appendChild(td);
    var div=document.createElement('div');
    div.style.fontSize="18px"
    div.appendChild(td);
    //div.setAttribute('class','grow');
    tr.cells[0].appendChild( div );
    //tr.cells[1].appendChild( document.createTextNode('Text2') );
    tr.cells[0].style.border='none';
    tr.cells[0].style.width='190px';
   tr.cells[0].style.height='40px';
    var td2=document.createElement('td');
    tr.appendChild(td2);
    td2.setAttribute('text-align','left');
    td2.setAttribute('vertical-align','top');
    td2.setAttribute('align','left');
    //tr.appendChild( document.createElement('td') );
    //var h1 = document.createElement("mark")
    //j=Math.floor(Math.random()*4)+1;
    //h1.style.backgroundColor=arr[j-1];
    //h1.style.borderRadius = "5px";
    //j=Math.floor(Math.random()*4)+1;
   // h1.style.color=arr[j-1];
    var td3 =document.createTextNode(params[i+5][1]);
     //h1.appendChild(td3);
    var div2=document.createElement('div');
    div2.style.fontSize="18px"
    div2.appendChild(td3);
    //div.setAttribute('class','grow');
    tr.cells[1].appendChild( div2 );
    //tr.cells[1].appendChild( document.createTextNode('Text2') );
    tr.cells[1].style.border='none';
    tr.cells[1].style.width='190px';
    tr.cells[1].style.height='40px';
     var t1d2=document.createElement('td');
    tr.appendChild(t1d2);
    t1d2.setAttribute('text-align','left');
    t1d2.setAttribute('vertical-align','top');
    t1d2.setAttribute('align','left');
    //tr.appendChild( document.createElement('td') );
    //var h2 = document.createElement("mark")
    //j=Math.floor(Math.random()*4)+1;
    //h2.style.backgroundColor=arr[j-1];
    //h2.style.borderRadius = "5px";
    //j=Math.floor(Math.random()*4)+1;
   // h2.style.color=arr[j-1];
    var t1d3 =document.createTextNode(params[i+20][1]);
     //h2.appendChild(t1d3);
    var di1v2=document.createElement('div');
    di1v2.style.fontSize="18px"
    di1v2.appendChild(t1d3);
    tr.style.height='10px';
    //div.setAttribute('class','grow');
    tr.cells[2].appendChild( di1v2 );
    //tr.cells[1].appendChild( document.createTextNode('Text2') );
    tr.cells[2].style.border='none';
    tr.cells[2].style.width='190px';
    tr.cells[2].style.height='40px';




    table.appendChild(tr);
}
tablearea2.appendChild(table);
           var tablearea = document.getElementById('listtable');
    table = document.createElement('table');


for (var i = 0; i < 6 ; i++) {
    var tr = document.createElement('tr');
    var td1=document.createElement('td');
    tr.appendChild(td1);
    td1.setAttribute('text-align','left');
    td1.setAttribute('vertical-align','top');
    td1.setAttribute('align','left');
    //tr.appendChild( document.createElement('td') );

    var td =document.createTextNode(params[i][3]);
    var div=document.createElement('div');
    div.appendChild(td);
    div.setAttribute('class','grow');
    tr.cells[0].appendChild( div );
    //tr.cells[1].appendChild( document.createTextNode('Text2') );
    tr.cells[0].style.border='1px solid black';
    tr.cells[0].style.width='700px';
    tr.cells[0].style.height='20px';



    table.appendChild(tr);
}
tablearea.appendChild(table);
//alert(params[0])
feedback=[];
var stage = new Konva.Stage({
        container: 'container',
        width:750,
        height: 700
      });


      var layer = new Konva.Layer();

      var arc1 = new Konva.Circle({
        x:350 ,
        y:350,
        radius:60,
        stroke: 'black',
        strokeWidth: 2,
        //fill:"#ff8080"
        fill:'red'

      });
      var user = new Konva.Text({
        x: 330,
        y: 340,
        fontFamily: 'Calibri',
        fontSize: 24,
        text: 'User',
        fill: 'black'
      });
      //user.text("User");
      layer.add(arc1);
      layer.add(user);
      layer.draw();
     // arc1.on('click',function(){
       // this.fill("red");
        //layer.draw();
        //stage.destroyChildren();
        //canvas();

      //});
      //layer.add(arc1);
      var text = new Konva.Text({
        x: 0,
        y: 0,
        fontFamily: 'Calibri',
        fontSize: 14,
        text: '',
        fill: 'black'
      });
      layer.add(text);
      var arc2 = new Konva.Arc({
        x:350 ,
        y:350,
        innerRadius: 139,
        outerRadius: 140,
        angle:270 ,
        stroke: 'black',
        strokeWidth: 2,
        rotation:135
      });
      layer.add(arc2);
      var arc3 = new Konva.Arc({
        x:350 ,
        y:350,
        innerRadius: 239,
        outerRadius: 240,
        angle:270 ,
        stroke: 'black',
        strokeWidth: 2,
        rotation:135
      });



      layer.add(arc3);


      //abhi tak bas arc bana hai

      var keys=["computer vision","acd","abc"];
      var x=[420,500,460];
      var y=[420,520,470];
      var count=[0,0,0];
      var labels=["a1","a2","a3"];
      //var feedback=[];

      var i;
      //alert(params.length);
      for(i=0;i<params.length;i++)
      {
      var x_cord=300;
      var y_cord=300;
         var a=Math.floor((Math.random() * 180)) + 120;
         var b=Math.floor((Math.random() * 180)) + 420;
         var k=Math.round(Math.random());
         if(k==0)
         {
            x_cord=a;
         }
         else
         {
            x_cord=b;
         }
         a=Math.floor((Math.random() * 380)) + 120;
         b=Math.floor((Math.random() * 380)) + 120;
         k=Math.round(Math.random());
         if(k==0)
         {
            y_cord=a;
         }
         else
         {
            y_cord=b;
         }
         //alert(x);
         //alert(y);
         var arr=['#ECA43E','#EF5B17','#B5D57C','#0066ff','#990000'];
         var j=Math.floor(Math.random()*4)+1;
         var arck = new Konva.Circle({
        x:x_cord ,
        y:y_cord,
        radius:3,
        stroke: arr[j-1],
        strokeWidth: 2,
        fill:arr[j-1],


      });
      arck.addName(params[i][1]);
      arck.on('mouseover', function() {
       /* var text = new Konva.Text({
        x: 10,
        y: 10,
        fontFamily: 'Calibri',
        fontSize: 24,
        text: '',
        fill: 'black'
      });
      layer.add(text);*/
        writeMessage(text,layer,this.x(),this.y(),this.name());
      });
     arck.on('mouseout', function() {
        writeMessage(text,layer,"");
      });
      layer.add(arck);
      }
      for(i=0;i<10;i++)
      {
         var x_cord=450;
      var y_cord=450;
         var a=Math.floor((Math.random() * 180)) + 120;
         var b=Math.floor((Math.random() * 180)) + 350;
         var k=Math.round(Math.random());
         if(k==0)
         {
            x_cord=a;
         }
         else
         {
            x_cord=b;
         }
         a=Math.floor((Math.random() * 380)) + 120;
         b=Math.floor((Math.random() * 380)) + 120;
         k=Math.round(Math.random());
         if(k==0)
         {
            y_cord=a;
         }
         else
         {
            y_cord=b;
         }
       labels[i]  = new Konva.Label({
        x: x_cord,
        y: y_cord,
        opacity: 0.6

      });
      var arr=['#33d6ff','#ffffb3','#fff0e6','#ffe6ff','#dfff80','#ffcc66'];
         var j=Math.floor(Math.random()*arr.length);
      labels[i].add(
        new Konva.Tag({
          fill: arr[j],
          lineJoin: 'round',
          shadowColor: 'red',
          shadowBlur: 10,
          shadowOffset: 10,
          shadowOpacity: 0.5,
          cornerRadius: 5
         /* pointerDirection: 'down',
          pointerWidth: 0,
          pointerHeight: 0,
          lineJoin: 'round',
          shadowColor: 'red',
          shadowBlur: 10,
          shadowOffset: 10,
          shadowOpacity: 0.5*/
        })
      );

      labels[i].add(
        new Konva.Text({
          text: params[i+10][1],
          fontFamily: 'Calibri',
          fontSize: 16,
          padding: 5,
          fill: 'black',
        })
      );
      //var ku=["kundan"];
      /*labels[i].on('click', function() {
        //var j=i;
        //promt(typeof keys[i]);
        //prompt("hi"+keys[j]);
        alert(i);
      });*/
      layer.add(labels[i]);
    }
    labels[0].on('click', function() {
        //var j=i;
        //promt(typeof keys[i]);
        //prompt("hi"+keys[j]);
        //alert(keys[0]);
        this.getText().fill('green')
        layer.draw();
        feedback.push(params[10][0]);
      });
    labels[1].on('click', function() {
        //var j=i;
        //promt(typeof keys[i]);
        //prompt("hi"+keys[j]);
        //alert(keys[1]);
        this.getText().fill('green')
        layer.draw();
        feedback.push(params[11][0]);
      });
    labels[2].on('click', function() {
        //var j=i;
        //promt(typeof keys[i]);
        //prompt("hi"+keys[j]);
        //alert(keys[2]);
        this.getText().fill('green')
        layer.draw();
        feedback.push(params[12][0]);
      });
      labels[3].on('click', function() {
      this.getText().fill('green')
        layer.draw();
      feedback.push(params[13][0]);
      });
      labels[4].on('click', function() {
      this.getText().fill('green')
        layer.draw();
      feedback.push(params[14][0]);
      });
      labels[5].on('click', function() {
      this.getText().fill('green')
        layer.draw();
      feedback.push(params[15][0]);
      });
      labels[6].on('click', function() {
      this.getText().fill('green')
        layer.draw();
      feedback.push(params[16][0]);
      });
      labels[7].on('click', function() {
      this.getText().fill('green')
        layer.draw();
      feedback.push(params[17][0]);
      });
      labels[8].on('click', function() {
      this.getText().fill('green')
        layer.draw();
      feedback.push(params[18][0]);
      });
      labels[9].on('click', function() {
      this.getText().fill('green')
        layer.draw();
      feedback.push(params[19][0]);
      });

      console.log(feedback);
      stage.add(layer);





 }

