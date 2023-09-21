(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('dragMainWinder', ['$document', dragMainWinderDirective]);

    function dragMainWinderDirective($document) {
        return {
            scope: { position: '='}, 
            link: function (scope, element, attr) {
                var startY = 0, startX=0, y=0;
                var centerX=0;
                var clientRect; 
                element.on('mousedown', function (event) {
                    var container = document.getElementsByClassName('main-control-container');
                    clientRect = container[0].getBoundingClientRect();
                    centerX = clientRect.left + (clientRect.width/2);
                    event.preventDefault();
                    element.addClass('on');
                    y = clientRect.height-(scope.position+0.5)*clientRect.height;
                    startY = event.screenY - y; 
                    startX = event.screenX - centerX;
                    if(startX<15 && startX>-15){
                        $document.on('mousemove', mousemove);
                        $document.on('mouseup', mouseup);
                    }
                });
                function mousemove(event) {
                    y = event.screenY - startY;
                    if (element.hasClass('on')) {
                        if (y >= 0 && y <= clientRect.height) { 
                            scope.position = 0.5-y/clientRect.height;
                            scope.$apply();
                            element.css({
                                top: y/clientRect.height*100 + '%'
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
