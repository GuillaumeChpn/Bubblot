(function () {
    'use strict';
    angular
        .module('bubblot')
        .directive('dragWinder', ['$document', dragWinderDirective]);

    function dragWinderDirective($document) {
        return {
            scope: {position: '='}, 
            link: function (scope, element, attr) {
                var startY = 0, y=0, clientRect;
                element.on('mousedown', function (event) {
                    var container = document.getElementsByClassName('control-container');
                    clientRect = container[0].getBoundingClientRect();
                    event.preventDefault();
                    element.addClass('on');
                    y = clientRect.height-(scope.position+0.5)*clientRect.height;
                    startY = event.screenY - y; 
                    $document.on('mousemove', mousemove);
                    $document.on('mouseup', mouseup);
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
