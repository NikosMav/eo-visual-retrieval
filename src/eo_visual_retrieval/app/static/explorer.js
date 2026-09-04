// The grouped native picker also works without JavaScript.
const picker = document.getElementById('item_id');
const classes = document.getElementById('scene-class');
const groups = Array.from(picker.children).map(group => group.cloneNode(true));
function filterQueries() {
  const previous = picker.value;
  picker.replaceChildren(...groups.filter(group => !classes.value || group.label === classes.value).map(group => group.cloneNode(true)));
  if (Array.from(picker.options).some(option => option.value === previous)) picker.value = previous;
}
document.getElementById('class-filter').hidden = false;
classes.addEventListener('change', filterQueries);
filterQueries();
