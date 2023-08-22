(function () {
    'use strict';
    angular
        .module('bubblot')
        .controller('messageCtrl', messageCtrl);

    function messageCtrl($scope, $element) {
        //Structure of the HTML file that is sent to the client
        var htmlContent1 = "<!DOCTYPE html>\r\n";
        htmlContent1 += "<html>\r\n";
        htmlContent1 += "<head>\r\n";
        htmlContent1 += "<meta charset='utf-8'>\r\n";
        htmlContent1 += "<title>Bubblot</title>\r\n";
        htmlContent1 += "<link href='main.css' type='text/css' rel='stylesheet'>\r\n";
        htmlContent1 += "</head>\r\n";
        htmlContent1 += "<body style='background-color: black;overflow:hidden;'>\r\n";
        var htmlContent2 = "</body>\r\n";
        htmlContent2 += "<script type='text/javascript' src='main.js'></script>\r\n";
        htmlContent2 += "</html>\r\n";
        var htmlContent = null;
        var messageText = document.getElementsByClassName("message-text");
        //Function to call when modifying the input
        for (var i = 0; i < messageText.length; i++) {
            if(i==0){
                messageText[i].addEventListener('keyup', updateTargetDown);
            }
            else{
                messageText[i].addEventListener('keyup', updateTargetUp);
            }
        }
        //Update top input when bottom input is modified + send message when enter
        function updateTargetUp(event) {
            if(event.keyCode==13){
                messageText[0].value=messageText[1].value;
                htmlContent = htmlContent1 + "<h1 class='message'>" + messageText[1].value + "</h1>\r\n" + htmlContent2;
                fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                    if (err) throw err;
                    else {
                        ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                        })
                    }
                })
            }
            else{
                messageText[0].value=messageText[1].value;
            }
        }
        //Update bottom input when top input is modified + send message when enter
        function updateTargetDown(event) {
            if(event.keyCode==13){
                messageText[1].value=messageText[0].value;
                htmlContent = htmlContent1 + "<h1 class='message'>" + messageText[0].value + "</h1>\r\n" + htmlContent2;
                fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                    if (err) throw err;
                    else {
                        ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                        })
                    }
                })
            }
            else{
                messageText[1].value=messageText[0].value;
            }
        }
        //Function to display/hide emoticons
        $scope.displayListEmoFunc = function (value) {
            var emoticonsList = $element.find(".emoticons-list");
            if (value) emoticonsList.show();
            else emoticonsList.hide();
        }
        //Function to write HTML file with happy emoticon + refresh the web page on the client
        $scope.printHappy = function () {
            htmlContent = htmlContent1 + "<div class='message-image-happy'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with sad emoticon + refresh the web page on the client
        $scope.printSad = function () {
            htmlContent = htmlContent1 + "<div class='message-image-sad'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with nice emoticon + refresh the web page on the client
        $scope.printNice = function () {
            htmlContent = htmlContent1 + "<div class='message-image-nice'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with minion emoticon + refresh the web page on the client
        $scope.printMinionFace = function () {
            htmlContent = htmlContent1 + "<div class='message-image-minion-face'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
        }
        //Function to write HTML file with minion emoticon + refresh the web page on the client
        $scope.printMinionGlass = function () {
            htmlContent = htmlContent1 + "<div class='message-image-minion-glass'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with minion emoticon + refresh the web page on the client
        $scope.printMinionGlass2 = function () {
            htmlContent = htmlContent1 + "<div class='message-image-minion-glass2'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with pacman emoticon + refresh the web page on the client
        $scope.printPacman = function () {
            htmlContent = htmlContent1 + "<div class='message-image-pacman'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with hublot emoticon + refresh the web page on the client
        $scope.printHublot = function () {
            htmlContent = htmlContent1 + "<div class='message-image-hublot'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with shark emoticon + refresh the web page on the client
        $scope.printShark = function () {
            htmlContent = htmlContent1 + "<div class='message-image-shark'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with minion emoticon + refresh the web page on the client
        $scope.printMinion = function () {
            htmlContent = htmlContent1 + "<div class='message-image-minion'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with minion emoticon + refresh the web page on the client
        $scope.printMinion2 = function () {
            htmlContent = htmlContent1 + "<div class='message-image-minion2'></div>\r\n" + htmlContent2;
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
        //Function to write HTML file with bubble emoticon + refresh the web page on the client
        $scope.printBubble = function () {
            htmlContent = htmlContent1 + "<div class='message-image-bubble'></div>\r\n" + htmlContent2;   
            fs.writeFile('Raspberry/raspberry-web/index.html', htmlContent, function (err) {
                if (err) throw err;
                else {
                    ssh.execCommand('/home/pi/Desktop/refresh.sh', { cwd: '/var/www' }).then(function (result) {
                    })
                }
            });
            $scope.displayListEmo=false;
        }
    }
}());
