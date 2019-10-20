BASE_URL =  'http://127.0.0.1:5000';
labelMap = {};
color = ["#27446E","#0066FF","#29A9FF","#91C6FF","#717C7E"];
display_limit = 4;

getMap();
// get label map
function getMap() {
    let url =`${BASE_URL}/label_map`;
    $.get(url, function (result) {
        labelMap = result;
    })
}

// generate document Query Url
function generateQueryUrl(text) {
    return `${BASE_URL}/top_labels?document=${text}`;
}


function onSearch(){
    let text = $("#document").val();
    let url = generateQueryUrl(text);
    let resultList = $('#resultList');
    resultList.empty(); // remove old children

    $.get(url, function (result) {
        console.log(result);
        console.log(labelMap);
        $("#result").removeClass("hide");
        let index = 1;
        let scores = [], labels=[];
        let sum = 0;


        for(let i = 0; i < result.length; i++){
            let score = result[i]["score"];
            score= parseFloat(score.toFixed(4));
            scores.push(score);
            sum += score;
            let label = result[i]["label"];
            label = labelMap[label];
            labels.push(label);

            let list_item = `<li class="list-group-item">The top ${index}
                            label is <b><i>${label}</i></b> with <b>possibility</b> = <b>${score}</b></li>`;
            resultList.append($(list_item));
            index ++;
            if(index > display_limit){
                break;
            }


        }
        let other_num = 1 - sum;
        scores.push(other_num);
        labels.push("Others");
        console.log(scores, labels);
        drawCircle("circle", scores, color, labels);
    })
}

function clearCanvas()
{
    let c=document.getElementById("circle");
    let cxt=c.getContext("2d");
    c.height=c.height;
}


function drawCircle(canvasId, data_arr, color_arr, text_arr)
{
    clearCanvas();
    let c = document.getElementById(canvasId);
    let ctx = c.getContext("2d");

    let radius = c.height / 2 - 20; //radius
    let ox = radius + 20, oy = radius + 20; //circle

    let width = 30, height = 10; //
    let posX = ox * 2 + 20, posY = 30;   //
    let textX = posX + width + 5, textY = posY + 10;

    let startAngle = 0;
    let endAngle = 0;
    for (let i = 0; i < data_arr.length; i++)
    {
        endAngle = endAngle + data_arr[i] * Math.PI * 2;
        ctx.fillStyle = color_arr[i];
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.arc(ox, oy, radius, startAngle, endAngle, false);
        ctx.closePath();
        ctx.fill();
        startAngle = endAngle;

        ctx.fillStyle = color_arr[i];
        ctx.fillRect(posX, posY + 20 * i, width, height);
        ctx.moveTo(posX, posY + 20 * i);
        ctx.font = 'bold 15px';
        let percent = text_arr[i];
        ctx.fillText(percent, textX, textY + 20 * i);
    }
}

