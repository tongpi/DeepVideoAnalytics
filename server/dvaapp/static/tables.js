function InitializeTables(){
    $('.dataTables').dataTable({
        responsive: true,
		language:{url: "chinese.json"}   
    });
    $('.dataTables-dict').dataTable({
        responsive: true,
        "bPaginate": false,
		language:{url: "chinese.json"}   
    });
    $('.dataTables-nofilter').dataTable({
        responsive: true,
        "bPaginate": false,
        "bFilter": false,
		language:{url: "chinese.json"} 
    });
}
