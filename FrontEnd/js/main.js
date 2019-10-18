function onSearch(){
    let text = $("#document").val();
    let url = `http://35.208.155.164:6516/top_labels?document=${text}`;

    $.get(url, function (result) {
        console.log(result);
        $("#result").removeClass("hide");
        let index = 1;
        for(let entry in result){
            let score = result[entry]["score"];
            score= score.toFixed(4);
            let label = result[entry]["label"];
            let sentence = `The top ${index} label is "${label}" with score ${score}`;
            $(`#top${index}`).text(sentence);
            index ++;
        }
    })
}