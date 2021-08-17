var ret;
var selectedTab = "Graphs";

function getCookieVal(name) {
    var cookie = decodeURIComponent(document.cookie);
    var vals = cookie.split(';');
    name += '=';
    var ret;
    vals.forEach(val => {
        while (val.charAt(0) == ' ') {
            val = val.substring(1);
        }
        if (val.indexOf(name) == 0) {
            ret = val.substring((name).length, val.length);
        } else {
            ret = '';
        }
    });
    return ret;
}

const csrftoken = getCookieVal('csrftoken');

function generateTable() {
    if (ret.data !== undefined && ret.data !== null && !jQuery.isEmptyObject(ret.data)) {
        var cols = [];
        for (const col in ret.columns) {
            cols.push({title: ret.columns[col]});
        }

        $('#table').DataTable( {
            pagingType: "full_numbers",
            data: ret.data,
            columns: cols
        } );
    } else {
        $('#table_error').html("There are no lists provided by this pipeline.");
        $('#table_error').show();
        $("#table").hide();
    }
}


function generateText() {
    try {
        if (ret.text){
            $("#text").html(ret.text);
        }
        else {
            $("#text").html("There is no text provided by this pipeline");
        }
    } catch (error) {
        $("#text").html("There is no text provided by this pipeline");
    }
}


function repaint(){
            if (selectedTab == "Graphs"){
                $('#table_wrapper').hide();
                $("#text").hide();
                $('#graphs').show();
            }
            else if (selectedTab == "Lists"){
                $('#graphs').hide();
                $("#text").hide();
                $('#table_wrapper').show();
            }
            else {
                $('#graphs').hide();
                $('#table_wrapper').hide();
                $("#text").show();
            }
}


function paint(){
            repaint();
            $('#wait').hide();
            $('#warpper').show();
            if (ret.graphs !== undefined && ret.graphs !== null) {
            var graphs = "";
                var container = document.getElementById('graphs');
                //alert(({}).toString.call(ret.graphs).match(/\s([a-zA-Z]+)/)[1].toLowerCase());
                for (var i = 0; i < ret.graphs.length; i++) {
                    var gid = "createdGraph_" + i;
                    var div = document.createElement('div');
                    graphs += ret.graphs[i];
                }
                $('#graphs').html(graphs);
                $('#graphs').show();
            }
            else {
                $('#graphs').html("There are no graphs provided by this pipeline.");
            }
            generateTable();
            generateText();

}


function getParams(){
    pipeline = $('#id_pipelines').val();
    type = $('#id_type').val();
    var params = {
        "select_pipeline": 1,
        "pipeline": pipeline,
        "type": type
    };
    $.ajax({
        type: 'POST',
        url: '.',
        beforeSend: function(request){
            /* eslint-disable no-undef */
            request.setRequestHeader('X-CSRFToken', csrftoken);
            /* eslint-enable no-undef */
        },
        data: jQuery.param(params),
        success: function(data){
            if (data !== undefined && data !== null){
                $("#attrs").html(data.userform);
            }
            else{
                $("#attrs").html("");
            }
        },
        error: function(request){
                var response = JSON.parse(request.responseText);
                $('#loader').hide();
                console.log(response.messages);
                for (var key in response.messages) {
                    if(!Object.prototype.hasOwnProperty.call(response.messages, key)){
                        continue;
                    }
                    $('#id_' + key).addClass('is-invalid');
                    $('#page_content').append('<div class="alert alert-danger" role="alert">' + response.messages[key] + '</div>');
                }
            }
        });
}


$(document).ready(function(){
    $('#table_wrapper').hide();
    $('#graphs').hide();
    $('#text').hide();
    $('#warpper').hide();
    $('#pipeline_results').hide();
    $('#loader').hide();
    $('#pipeline_select').show();
    $('#wait').show();

    $('#submit_btn').click(function(event) {
        event.preventDefault();
        $('#loader').show();
        $('#errdiv').html("");

        var elements = document.getElementById("attrs").querySelectorAll('input', 'textarea');
        var files = document.getElementById("attrs").querySelectorAll('select');
        var params = {
            "type": document.getElementById("type").elements[1].value,
            "pipelines": document.getElementById("pipelines").elements[1].value,
            "input": document.getElementById("files").elements[1].value,
        };
        if (elements != null && elements.length > 0) {
            for (var i = 0; i < elements.length; i++) {
                params[elements[i].getAttribute("name")] = elements[i].value;
            }
        }
        if (files != null && files.length > 0) {
            for (var i = 0; i < files.length; i++) {
                params[files[i].getAttribute("name")] = files[i].options[files[i].selectedIndex].text;
            }
        }

        type_obj = document.getElementById("type");

        $.ajax({
            type: 'POST',
            url: '.',
            beforeSend: function(request){
                /* eslint-disable no-undef */
                request.setRequestHeader('X-CSRFToken', csrftoken);
                /* eslint-enable no-undef */
            },
            data: jQuery.param(params),//$(this).serialize(),
            success: function(data){
                $('#loader').hide();
                if (data !== undefined && data !== null){
                    if (!Object.prototype.hasOwnProperty.call(data, 'empty')) {
                        if (data.hasOwnProperty('error')) {
                            $('#main_content').show();
                            $('#errdiv').show();
                            var hr = document.createElement("hr");
                            var errtag = document.createElement("p");
                            errtag.innerHTML = "<b>Message: </b>";
                            var text = document.createTextNode(data.error_msg);
                            errtag.appendChild(text);
                            var errinfo = document.createElement("p");
                            errinfo.innerHTML = "<b>Error info: </b>"
                            var err = document.createTextNode(data.error_info);
                            errinfo.appendChild(err);
                            alert("Something went wrong");
                            $('#errdiv').append(hr);
                            $('#errdiv').append(errtag);
                            $('#errdiv').append(errinfo);
                        } else {
                            if (data.hasOwnProperty('train')) {
                                $('#errdiv').show();
                                $('#errdiv').html(data.train);
                            }
                            else {
                                $('#attrs').hide();
                                $('#pipeline_results').show();
                                $('#pipeline_select').hide();
                                ret = data;
                                paint();
                            }
                        }
                    }
                    else {
                        $('#main_content').html('No updates returned for this specific query. Try a different query.');
                    }
                }
                else{
                    $('#main_content').html('Invalid Data Returned by Backend');
                }
            },
            error: function(request){
                var response = JSON.parse(request.responseText);
                $('#loader').hide();
                console.log(response.messages);
                for (var key in response.messages) {
                    if(!Object.prototype.hasOwnProperty.call(response.messages, key)){
                        continue;
                    }
                    $('#id_' + key).addClass('is-invalid');
                    $('#page_content').append('<div class="alert alert-danger" role="alert">' + response.messages[key] + '</div>');
                }

            }
        });
    });
    
    $('input[type=radio][name="group"]').change(function() {
        selectedTab = $(this).val();
        //$('#test').html($(this).val());
        repaint();
    });

    $('#id_type').change(getParams);
    $('#id_pipelines').change(getParams);

    setTimeout(function(){
        $('#submit_btn').click();
    }, 500);

});