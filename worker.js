chrome.action.onClicked.addListener(tab => chrome.tabs.create({
  url: '/data/window/index.html',
  index: tab.index + 1
}));
