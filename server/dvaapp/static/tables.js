function InitializeTables(){
    $('.dataTables').dataTable({
        responsive: true,
		language:{url: "/static/chinese.json"}   
    });
    $('.dataTables-dict').dataTable({
        responsive: true,
        "bPaginate": false,
		language:{url: "/static/chinese.json"}   
    });
    $('.dataTables-nofilter').dataTable({
        responsive: true,
        "bPaginate": false,
        "bFilter": false,
		language:{url: "/static/chinese.json"} 
    });
}
