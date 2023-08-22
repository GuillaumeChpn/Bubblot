(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('drag3dView', ['$document', drag3dViewDirective]);

    function drag3dViewDirective($document) {
        return {
            scope: { position: '='}, 
            link: function (scope, element, attr) {
                var startY = 0, y=0, clientRect; 
                element.on('mousedown', function (event) {
                    var cadran=document.getElementsByClassName('cadrans-container view');
                    clientRect = cadran[0].getBoundingClientRect();
                    event.preventDefault();
                    element.addClass('on');
                    y = clientRect.height*(70-35*Math.tan((90-scope.position)*Math.PI/180))/100; 
                    startY = event.screenY - y; 
                    $document.on('mousemove', mousemove);
                    $document.on('mouseup', mouseup);
                });

                function mousemove(event) {
                    y = event.screenY - startY;
                    if (element.hasClass('on')) {
                        if (y/clientRect.height*100 >= 9.3 && y/clientRect.height*100 <= 35) { 
                            scope.position = 90-Math.atan(((y/clientRect.height*100)-70)/(-35))*180/Math.PI; 
                            scope.$apply();
                            element.css({
                                top: y/clientRect.height*100 + '%'//slider start at top=10% and ends at top=64% 
                            });
                        }
                    }
                    else {
                        element.css({
                            top: 0
                        });
                    } 
                }

                function mouseup() {
                    $document.off('mousemove', mousemove);
                    $document.off('mouseup', mouseup);
                }
            }
        }
    }
} ());
