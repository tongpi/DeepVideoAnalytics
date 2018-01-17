function InitializeTables(){
    $('.dataTables').dataTable({
        responsive: true,
		language:{url: "Chinese.json"}   
    });
    $('.dataTables-dict').dataTable({
        responsive: true,
        "bPaginate": false,
		language:{url: "Chinese.json"}   
    });
    $('.dataTables-nofilter').dataTable({
        responsive: true,
        "bPaginate": false,
        "bFilter": false,
		language:{url: "Chinese.json"} 
    });
}
