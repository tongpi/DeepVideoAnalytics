function InitializeTables(){
    $('.dataTables').dataTable({
        responsive: true,
		language:{  
				  "sProcessing": "处理中...",  
				  "sLengthMenu": "显示 _MENU_ 条记录",  
				  "sZeroRecords": "没有匹配记录",  
				  "sInfo": "显示第 _START_ 到 _END_ 条记录，共 _TOTAL_ 条",  
				  "sInfoEmpty": "显示第 0 到 0 条记录，共 0 条",  
				  "sInfoFiltered": "(由 _MAX_ 条记录过滤)",  
				  "sInfoPostFix": "",  
				  "sSearch": "搜索:",  
				  "sUrl": "",  
				  "sEmptyTable": "暂无信息显示",  
				  "sLoadingRecords": "载入中...",  
				  "sInfoThousands": ",",  
				  "oPaginate": {  
					  "sFirst": "首页",  
					  "sPrevious": "上页",  
					  "sNext": "下页",  
					  "sLast": "末页" 
				  },  
				  "oAria": {  
					  "sSortAscending": ": 以升序排列此列",  
					  "sSortDescending": ": 以降序排列此列"  
				  }  
				}   
    });
    $('.dataTables-dict').dataTable({
        responsive: true,
        "bPaginate": false,
		language:{  
				  "sProcessing": "处理中...",  
				  "sLengthMenu": "显示 _MENU_ 条记录",  
				  "sZeroRecords": "没有匹配记录",  
				  "sInfo": "显示第 _START_ 到 _END_ 条记录，共 _TOTAL_ 条",  
				  "sInfoEmpty": "显示第 0 到 0 条记录，共 0 条",  
				  "sInfoFiltered": "(由 _MAX_ 条记录过滤)",  
				  "sInfoPostFix": "",  
				  "sSearch": "搜索:",  
				  "sUrl": "",  
				  "sEmptyTable": "暂无信息显示",  
				  "sLoadingRecords": "载入中...",  
				  "sInfoThousands": ",",  
				  "oPaginate": {  
					  "sFirst": "首页",  
					  "sPrevious": "上页",  
					  "sNext": "下页",  
					  "sLast": "末页" 
				  },  
				  "oAria": {  
					  "sSortAscending": ": 以升序排列此列",  
					  "sSortDescending": ": 以降序排列此列"  
				  }  
				}   
    });
    $('.dataTables-nofilter').dataTable({
        responsive: true,
        "bPaginate": false,
        "bFilter": false,
		language:{  
				  "sProcessing": "处理中...",  
				  "sLengthMenu": "显示 _MENU_ 条记录",  
				  "sZeroRecords": "没有匹配记录",  
				  "sInfo": "显示第 _START_ 到 _END_ 条记录，共 _TOTAL_ 条",  
				  "sInfoEmpty": "显示第 0 到 0 条记录，共 0 条",  
				  "sInfoFiltered": "(由 _MAX_ 条记录过滤)",  
				  "sInfoPostFix": "",  
				  "sSearch": "搜索:",  
				  "sUrl": "",  
				  "sEmptyTable": "暂无信息显示",  
				  "sLoadingRecords": "载入中...",  
				  "sInfoThousands": ",",  
				  "oPaginate": {  
					  "sFirst": "首页",  
					  "sPrevious": "上页",  
					  "sNext": "下页",  
					  "sLast": "末页" 
				  },  
				  "oAria": {  
					  "sSortAscending": ": 以升序排列此列",  
					  "sSortDescending": ": 以降序排列此列"  
				  }  
				}
    });
}
